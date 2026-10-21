#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <zstd.h>
#include <nvcomp/zstd.h>
#include <nvcomp/lz4.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/ans.h>
#include <nvcomp/bitcomp.h>
#include <nvcomp/cascaded.h>
using namespace std;
#define CK(x) do{auto e=(x); if(e!=cudaSuccess){printf("cuda %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
struct Col{string name; int w; vector<uint8_t> data;};
nvcompStatus_t* gSt;
double now(){return chrono::duration<double,milli>(chrono::steady_clock::now().time_since_epoch()).count();}
template<class T> T* dup(const vector<T>& v){T* d; CK(cudaMalloc(&d,max<size_t>(v.size(),1)*sizeof(T))); CK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return d;}
// Generic: Comp(inPtrs,inSizes,maxChunk,n,temp,tempBytes,outPtrs,outSizes,st), Decomp(...)
template<class CT, class CO, class DT, class DC, class MO>
void bench(const char* name, vector<Col>& cols, int chunkRows, CT compTemp, CO comp, MO maxOut, DT decTemp, DC dec, bool perType){
  size_t raw=0, compd=0;
  vector<const void*> allIn; vector<size_t> allInLen, allOutLen; vector<void*> allOut;
  vector<pair<string,double>> ratios;
  for(auto& c:cols){
    size_t rows=c.data.size()/c.w, nb=(rows+chunkRows-1)/chunkRows, cb=(size_t)chunkRows*c.w;
    uint8_t* d; CK(cudaMalloc(&d,c.data.size())); CK(cudaMemcpy(d,c.data.data(),c.data.size(),cudaMemcpyHostToDevice));
    size_t mo=0; maxOut(cb,c.w,&mo); mo=(mo+255)&~(size_t)255;
    uint8_t* o; CK(cudaMalloc(&o,mo*nb));
    vector<void*> ip(nb),op(nb); vector<size_t> is(nb);
    for(size_t b=0;b<nb;++b){ip[b]=d+b*cb; is[b]=min(cb,c.data.size()-b*cb); op[b]=o+b*mo;}
    size_t temp=0; compTemp(nb,cb,c.w,&temp,c.data.size()); void* t=nullptr; if(temp) CK(cudaMalloc(&t,temp));
    void** dip=dup(ip); size_t* dis=dup(is); void** dop=dup(op); size_t* dos; CK(cudaMalloc(&dos,nb*8));
    int s=comp((const void* const*)dip,dis,cb,nb,t,temp,(void* const*)dop,dos,c.w);
    CK(cudaDeviceSynchronize()); if(s){printf("%s compress status %d\n",name,s); return;}
    vector<size_t> os(nb); CK(cudaMemcpy(os.data(),dos,nb*8,cudaMemcpyDeviceToHost));
    size_t tot=0; for(auto x:os) tot+=x; raw+=c.data.size(); compd+=tot; ratios.push_back({c.name,(double)c.data.size()/tot});
    for(size_t b=0;b<nb;++b){allIn.push_back(op[b]); allInLen.push_back(os[b]); allOutLen.push_back(is[b]); void* x; CK(cudaMalloc(&x,is[b])); allOut.push_back(x);}
  }
  // one batched decompression of everything, and one of a single chunk (latency)
  auto run=[&](size_t n)->double{
    vector<const void*> ip(allIn.begin(),allIn.begin()+n); vector<size_t> il(allInLen.begin(),allInLen.begin()+n), ol(allOutLen.begin(),allOutLen.begin()+n); vector<void*> op(allOut.begin(),allOut.begin()+n);
    size_t mx=0,tot=0; for(auto x:ol) mx=max(mx,x),tot+=x;
    size_t temp=0; int ts=decTemp(n,mx,&temp,tot); if(ts) printf("%s dec temp status %d (n %zu mx %zu)\n",name,ts,n,mx); void* t=nullptr; if(temp) CK(cudaMalloc(&t,temp));
    auto dip=dup(ip); size_t* dil=dup(il); size_t* dol=dup(ol); void** dop=dup(op); size_t* das; CK(cudaMalloc(&das,n*8));
    double best=1e9;
    for(int it=0;it<5;++it){CK(cudaDeviceSynchronize()); double t0=now(); int s=dec((const void* const*)dip,dil,dol,das,n,t,temp,(void* const*)dop); CK(cudaDeviceSynchronize()); if(s){printf("%s dec status %d\n",name,s);return -1.0;} best=min(best,now()-t0);}
    return best;
  };
  double all=run(allIn.size()), one=run(1);
  printf("%-10s ratio %5.2f  batch of %zu chunks (%.0f MB): %7.2f ms = %5.1f GB/s   1 chunk: %.3f ms\n",name,(double)raw/compd,allIn.size(),raw/1e6,all,raw/all/1e6,one);
  printf("           per column:"); for(auto&[n,r]:ratios) printf(" %s %.1f", n.c_str(), r); printf("\n");
}
int main(int argc,char**argv){
  vector<Col> cols; const char* names[]={"WatchID","UserID","EventTime","ClientIP","CounterID","RegionID","ResolutionWidth","IsRefresh","EventDate","URL","SearchPhrase","Title","Referer"};
  for(auto n:names){Col c; c.name=n; FILE* f=fopen((string(argv[1])+"/"+n+".bin").c_str(),"rb"); fseek(f,0,SEEK_END); long l=ftell(f); fseek(f,0,SEEK_SET); c.data.resize(l); fread(c.data.data(),1,l,f); fclose(f); c.w=(string(n)=="WatchID"||string(n)=="UserID"||string(n)=="EventTime")?8:4; cols.push_back(move(c));}
  const int R=16384; CK(cudaMalloc(&gSt, 1<<16));
  { // CPU zstd level 1 and FOR+bitpack (reference)
    size_t raw=0,z=0,bp=0; ZSTD_CCtx* cc=ZSTD_createCCtx(); vector<uint8_t> o(ZSTD_compressBound(R*8));
    for(auto&c:cols){ size_t rows=c.data.size()/c.w; for(size_t r=0;r<rows;r+=R){ size_t n=min<size_t>(R,rows-r); z+=ZSTD_compressCCtx(cc,o.data(),o.size(),&c.data[r*c.w],n*c.w,1);
      long long mn=LLONG_MAX,mx=LLONG_MIN; for(size_t i=0;i<n;++i){long long v= c.w==8? ((long long*)c.data.data())[r+i] : ((int*)c.data.data())[r+i]; mn=min(mn,v); mx=max(mx,v);}
      unsigned long long range=(unsigned long long)mx-(unsigned long long)mn; int bits=range?64-__builtin_clzll(range):0; bp+=16+(n*bits+7)/8; } raw+=c.data.size(); }
    printf("CPU zstd-1 ratio %.2f   FOR+bitpack ratio %.2f\n",(double)raw/z,(double)raw/bp);
  }
  auto ty=[](int w){return w==8? NVCOMP_TYPE_LONGLONG: NVCOMP_TYPE_INT;};
#define DEC(algo) [&](size_t n,size_t mx,size_t* t,size_t tot){ return (int)nvcompBatched##algo##DecompressGetTempSizeAsync(n,mx,nvcompBatched##algo##DecompressDefaultOpts,t,tot);}, \
  [&](const void* const* ip,const size_t* il,const size_t* ol,size_t* as,size_t n,void* t,size_t tb,void* const* op){ return (int)nvcompBatched##algo##DecompressAsync(ip,il,ol,as,n,t,tb,op,nvcompBatched##algo##DecompressDefaultOpts,gSt,0);}
#define COMP(algo, OPTS) [&](size_t n,size_t cb,int w,size_t* t,size_t tot){ auto o=OPTS; return (int)nvcompBatched##algo##CompressGetTempSizeAsync(n,cb,o,t,tot);}, \
  [&](const void* const* ip,const size_t* is,size_t cb,size_t n,void* t,size_t tb,void* const* op,size_t* os,int w){ auto o=OPTS; return (int)nvcompBatched##algo##CompressAsync(ip,is,cb,n,t,tb,op,os,o,nullptr,0);}, \
  [&](size_t cb,int w,size_t* mo){ auto o=OPTS; return (int)nvcompBatched##algo##CompressGetMaxOutputChunkSize(cb,o,mo);}
  int w=4; (void)w;
  bench("zstd", cols, R, COMP(Zstd, nvcompBatchedZstdCompressDefaultOpts), DEC(Zstd), false);
  bench("lz4", cols, R, COMP(LZ4, nvcompBatchedLZ4CompressDefaultOpts), DEC(LZ4), false);
  bench("gdeflate", cols, R, COMP(Gdeflate, nvcompBatchedGdeflateCompressDefaultOpts), DEC(Gdeflate), false);
  bench("ans", cols, R, COMP(ANS, nvcompBatchedANSCompressDefaultOpts), DEC(ANS), false);
  bench("bitcomp", cols, R, COMP(Bitcomp, ([&]{auto o=nvcompBatchedBitcompCompressDefaultOpts; o.data_type=ty(w); return o;}())), DEC(Bitcomp), true);
  bench("cascaded", cols, R, COMP(Cascaded, ([&]{auto o=nvcompBatchedCascadedCompressDefaultOpts; o.type=ty(w); return o;}())), DEC(Cascaded), true);
}
