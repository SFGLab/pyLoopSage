"""Interactive 3D surface for the report: a small canvas viewer (drag to rotate, wheel to zoom, hover to read values).

No external libraries; the data are embedded as JSON so the report stays one self-contained file.
"""
import json

_JS = """
(function(){
const D=__DATA__, cv=document.getElementById('__ID__'), c=cv.getContext('2d');
const nx=D.x.length, ny=D.y.length, xr=[D.x[0],D.x[nx-1]], yr=[D.y[0],D.y[ny-1]];
let az=-0.95, el=0.62, zm=1, drag=null, hover=null;
const fmax=Math.max(...D.F.flat()), stops=['#0d0626','#3d1fa8','#6a3de8','#a98df0','#e0457b','#f58a3c','#ffd45e'];
function hex(h){return [1,3,5].map(i=>parseInt(h.substr(i,2),16));}
function col(t){t=Math.min(Math.max(t,0),1)*(stops.length-1);const i=Math.min(Math.floor(t),stops.length-2),f=t-i,a=hex(stops[i]),b=hex(stops[i+1]);
  return 'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')+')';}
function P(i,j,z){const X=2*(D.x[i]-xr[0])/(xr[1]-xr[0])-1, Y=2*(D.y[j]-yr[0])/(yr[1]-yr[0])-1, Z=z;
  const a=X*Math.cos(az)-Y*Math.sin(az), b=X*Math.sin(az)+Y*Math.cos(az);
  return {sx:a, sy:b*Math.cos(el)-Z*Math.sin(el), up:b*Math.sin(el)+Z*Math.cos(el)};}
function Pc(X,Y,Z){const a=X*Math.cos(az)-Y*Math.sin(az), b=X*Math.sin(az)+Y*Math.cos(az);
  return {sx:a, d:b*Math.cos(el)-Z*Math.sin(el), up:b*Math.sin(el)+Z*Math.cos(el)};}
function draw(){
  const w=cv.width, h=cv.height, S=Math.min(w,h*1.15)*0.25*zm, cx=w*0.46, cy=h*0.48;
  c.clearRect(0,0,w,h); c.fillStyle='#fff'; c.fillRect(0,0,w,h);
  const zf=f=>f/fmax*1.2-0.85, pts=[], quads=[];
  for(let j=0;j<ny;j++){pts.push([]);for(let i=0;i<nx;i++){const p=Pc(2*(D.x[i]-xr[0])/(xr[1]-xr[0])-1,2*(D.y[j]-yr[0])/(yr[1]-yr[0])-1,zf(D.F[j][i]));
    p.X=cx+p.sx*S; p.Y=cy-p.up*S; pts[j].push(p);}}
  for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){const q=[pts[j][i],pts[j][i+1],pts[j+1][i+1],pts[j+1][i]];
    quads.push({q:q,d:(q[0].d+q[1].d+q[2].d+q[3].d)/4,f:(D.F[j][i]+D.F[j][i+1]+D.F[j+1][i+1]+D.F[j+1][i])/4});}
  quads.sort((a,b)=>b.d-a.d);
  // floor frame
  const fl=[[-1,-1],[1,-1],[1,1],[-1,1]].map(p=>{const q=Pc(p[0],p[1],-0.85);return [cx+q.sx*S,cy-q.up*S];});
  c.strokeStyle='#b9b9c4'; c.lineWidth=1; c.beginPath(); fl.forEach((p,k)=>k?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1])); c.closePath(); c.stroke();
  // contour 'shadow' on the floor: filled levels plus contour lines (marching squares)
  const L=12, FP=(x,y)=>{const q=Pc(2*(x-xr[0])/(xr[1]-xr[0])-1,2*(y-yr[0])/(yr[1]-yr[0])-1,-0.85);return [cx+q.sx*S,cy-q.up*S];};
  for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){const f=(D.F[j][i]+D.F[j][i+1]+D.F[j+1][i+1]+D.F[j+1][i])/4;
    c.beginPath(); [[i,j],[i+1,j],[i+1,j+1],[i,j+1]].forEach((k,n)=>{const p=FP(D.x[k[0]],D.y[k[1]]);n?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]);});
    c.closePath(); c.fillStyle=col(Math.floor(f/fmax*L)/L); c.fill(); c.strokeStyle=c.fillStyle; c.lineWidth=0.6; c.stroke();}
  c.strokeStyle='rgba(17,17,17,.45)'; c.lineWidth=0.8;
  for(let l=1;l<L;l++){const t=l/L*fmax;
    for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){
      const v=[D.F[j][i],D.F[j][i+1],D.F[j+1][i+1],D.F[j+1][i]], xs=[D.x[i],D.x[i+1],D.x[i+1],D.x[i]], ys=[D.y[j],D.y[j],D.y[j+1],D.y[j+1]], pp=[];
      for(let e=0;e<4;e++){const a=e,b=(e+1)%4; if((v[a]-t)*(v[b]-t)<0){const u=(t-v[a])/(v[b]-v[a]); pp.push(FP(xs[a]+(xs[b]-xs[a])*u,ys[a]+(ys[b]-ys[a])*u));}}
      for(let k=0;k+1<pp.length;k+=2){c.beginPath(); c.moveTo(pp[k][0],pp[k][1]); c.lineTo(pp[k+1][0],pp[k+1][1]); c.stroke();}}}
  for(const o of quads){c.beginPath(); o.q.forEach((p,k)=>k?c.lineTo(p.X,p.Y):c.moveTo(p.X,p.Y)); c.closePath();
    c.globalAlpha=0.93; c.fillStyle=col(o.f/fmax); c.fill(); c.globalAlpha=1; c.strokeStyle='rgba(255,255,255,.55)'; c.lineWidth=0.5; c.stroke();}
  // axis labels on the two floor edges nearest to the viewer
  c.fillStyle='#111'; c.font='13px sans-serif'; c.textAlign='center';
  const edge=(a,b,txt,lo,hi)=>{const m=[(a[0]+b[0])/2,(a[1]+b[1])/2],o=Pc(m[0],m[1],-0.85),dx=m[0]*0.25,dy=m[1]*0.25,t=Pc(m[0]+dx,m[1]+dy,-0.85);
    c.fillText(txt,cx+t.sx*S,cy-t.up*S+18); c.fillStyle='#5a5a66'; c.font='11px sans-serif';
    const A=Pc(a[0]+(a[0]-b[0])*0.07,a[1]+(a[1]-b[1])*0.07,-0.85),B=Pc(b[0]+(b[0]-a[0])*0.07,b[1]+(b[1]-a[1])*0.07,-0.85); c.fillText(lo,cx+A.sx*S,cy-A.up*S+14); c.fillText(hi,cx+B.sx*S,cy-B.up*S+14); c.fillStyle='#111'; c.font='13px sans-serif';};
  const dep=(a,b)=>Pc((a[0]+b[0])/2,(a[1]+b[1])/2,-0.85).d;
  const ex=dep([-1,-1],[1,-1])<dep([-1,1],[1,1])?[[-1,-1],[1,-1]]:[[-1,1],[1,1]];
  const ey=dep([-1,-1],[-1,1])<dep([1,-1],[1,1])?[[-1,-1],[-1,1]]:[[1,-1],[1,1]];
  edge(ex[0],ex[1],D.xl,D.x[0].toFixed(0),D.x[nx-1].toFixed(0));
  edge(ey[0],ey[1],D.yl,D.y[0].toFixed(0),D.y[ny-1].toFixed(0));
  // colour bar
  const bx=w-70,by=h*0.2,bh=h*0.55; for(let k=0;k<bh;k++){c.fillStyle=col(k/bh);c.fillRect(bx,by+k,16,1.5);}
  c.fillStyle='#5a5a66'; c.textAlign='left'; c.font='11px sans-serif'; c.fillText('0',bx+22,by+8); c.fillText(fmax.toFixed(1),bx+22,by+bh); c.fillText('F (kT)',bx-4,by-8);
  if(hover){c.fillStyle='rgba(17,17,17,.88)'; const t=D.xl.split(' (')[0]+' '+D.x[hover.i].toFixed(0)+', '+D.yl.split(' (')[0]+' '+D.y[hover.j].toFixed(1)+', F '+D.F[hover.j][hover.i].toFixed(2)+' kT';
    c.font='12px sans-serif'; const tw=c.measureText(t).width+14; c.fillRect(hover.X+10,hover.Y-26,tw,22); c.fillStyle='#fff'; c.textAlign='left'; c.fillText(t,hover.X+17,hover.Y-10);}
  cv._pts=pts;
}
function pos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];}
cv.addEventListener('mousedown',e=>{drag=pos(e);}); window.addEventListener('mouseup',()=>{drag=null;});
cv.addEventListener('mousemove',e=>{const p=pos(e);
  if(drag){az+=(p[0]-drag[0])*0.008; el=Math.min(Math.max(el+(p[1]-drag[1])*0.006,0.05),1.5); drag=p; hover=null;}
  else{let best=null,bd=900;cv._pts.forEach((row,j)=>row.forEach((q,i)=>{const d=(q.X-p[0])**2+(q.Y-p[1])**2; if(d<bd){bd=d;best={i:i,j:j,X:q.X,Y:q.Y};}})); hover=best;}
  draw();});
cv.addEventListener('mouseleave',()=>{hover=null;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault(); zm=Math.min(Math.max(zm*(e.deltaY<0?1.1:0.9),0.5),3); draw();},{passive:false});
draw();
})();
"""


def surface_html(x, y, F, labels, uid="surf3d") -> str:
    """Canvas element plus script for the surface F[y, x]; labels = (x label, y label)."""
    data = dict(x=[round(float(v), 3) for v in x], y=[round(float(v), 3) for v in y],
                F=[[round(float(v), 3) for v in row] for row in F], xl=labels[0], yl=labels[1])
    js = _JS.replace("__DATA__", json.dumps(data)).replace("__ID__", uid)
    return (f'<canvas id="{uid}" width="1100" height="720" style="width:100%;max-width:1000px;display:block;margin:0 auto;'
            f'border:1px solid #ececf0;border-radius:14px;background:#fff;cursor:grab"></canvas>'
            f'<p class="caption">Drag to rotate, scroll to zoom, hover to read values.</p><script>{js}</script>')
