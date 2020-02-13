function [states, controls] = intial_conds(t_stat, t_contr)

intial_conds = load('/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat');
v = intial_conds.vres;
chi = intial_conds.chires;
gamma = intial_conds.gammares;
teta = intial_conds.tetares;
lam = intial_conds.lamres;
h = intial_conds.hres;
m = intial_conds.mres;
alfa = intial_conds.alfares;
delta = intial_conds.deltares;
t = intial_conds.t;

v_Int = pchip(t, v);
v_init = ppval(v_Int, t_stat);

chi_Int = pchip(t, chi);
chi_init = ppval(chi_Int, t_stat);

gamma_Int = pchip(t, gamma);
gamma_init = ppval(gamma_Int, t_stat);

teta_Int = pchip(t, teta);
teta_init = ppval(teta_Int, t_stat);

lam_Int = pchip(t, lam);
lam_init = ppval(lam_Int, t_stat);

h_Int = pchip(t, h);
h_init = ppval(h_Int, t_stat);

m_Int = pchip(t, m);
m_init = ppval(m_Int, t_stat);

alfa_Int_post = pchip(t, alfa);
delta_Int_post = pchip(t, delta);

alfa_init = [];
delta_init = [];
for i=1:length(t_contr(:,1))
    alfa_init = [alfa_init, ppval(alfa_Int_post, t_contr(i,:))];
    delta_init = [delta_init, ppval(delta_Int_post, t_contr(i,:))];
end
states = [v_init; chi_init; gamma_init; teta_init; lam_init; h_init; m_init];
controls = [alfa_init; delta_init];
end

