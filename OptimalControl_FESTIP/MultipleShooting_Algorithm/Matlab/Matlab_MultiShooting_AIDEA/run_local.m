obj = Spaceplane;
prob = Problem;
file = Files;

global bbb
bbb.f=1e16;
bbb.x=[];
bbb.xd=[];

nfev=1e5;
fname       = @(x)mask_MS(x, obj, prob, file);
load('bSol_5.mat')
bestmem=(bSol.xd - prob.LBV)./(prob.UBV - prob.LBV);
% bestmem=bSol.x;
XVmin=zeros(size(bestmem));
XVmax=ones(size(bestmem));
foptionsNLP=optimset('Display','iter-detailed','MaxFunEvals',nfev,'LargeScale','off','Algorithm','active-set','TolF',1e-14,'TolX',1e-14,'DiffMaxChange',1e-10);
[xgrad2,fvalgrad2,exitflag,output]=fmincon(fname,bestmem,[],[],[],[],XVmin,XVmax,[],foptionsNLP);
