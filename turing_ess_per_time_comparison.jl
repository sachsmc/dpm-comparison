using Turing
using CSV, DataFrames
using MCMCChains
using Turing: Variational
using Turing.RandomMeasures: stickbreak, DirichletProcess, StickBreakingProcess
using Optimisers

@model dp_model(y, Trunc) = begin
    n_obs = length(y)
    mu_star ~ filldist(Normal(0, 5), Trunc)
    sig_star = fill(0.3, Trunc)

    alpha ~ Gamma(1, 1)

    crm = DirichletProcess(alpha)
    v ~ filldist(StickBreakingProcess(crm), Trunc - 1)
    w = stickbreak(v)
    
    for i in 1:n_obs
        y[i] ~ UnivariateGMM(mu_star, sig_star, Categorical(w))
    end
end

sim_dat = CSV.read("data/sim_dat.csv", DataFrame)
z = sim_dat.z
y = sim_dat.y

Trunc = 20
mod = dp_model(y, Trunc)

n_samples = 28000
n_warmup = 4000
chain = sample(mod, MH(), n_samples + n_warmup, init_params = [-6, -2, 2, 6])

chain = chain[(n_warmup+1):(n_samples+n_warmup),:,:]

println(chain)

display(summarystats(chain))

q0 = Variational.meanfield(mod)
advi = ADVI(1, 2000)
q = vi(mod, advi, q0)

display(q)

nsamples = 10000
chain_vi = rand(q, nsamples)

display(summarystats(chain_vi))
