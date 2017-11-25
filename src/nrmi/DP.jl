export
    DP,
    logLevy,
    laplace,
    logGamma,
    logMeanMass,
    logMeanTotalMass,
    meanNumClusters

mutable struct DP <: NRMI # Dirichlet Process
  Ashape     ::  Float  # Gamma's shape parameter for alpha prior
  Ainvscale  ::  Float  # Gamma's inverse scale parameter for alpha prior
  alpha      ::  Float  # alpha parameter of DP's Lévy intensity measure
  logU       ::  Float  # Auxiliary random variable
end

DP(Ashape::Float, Ainvscale::Float) =
DP(Ashape, Ainvscale, (Ashape+1.0)/(Ainvscale+1.0), 0.0)

function logLevy(dp::DP, mass::Float, logu::Float)
  logLevy(mass, dp.alpha, 0., 1., logu)
end

function logLevy(mass::Float, alpha::Float, logu::Float)
  return log(alpha) - lgamma(1.0) - log(mass) - (exp(logu)+1)*mass
end

function laplace(dp::DP)
  return laplace(dp.alpha, dp.logU)
end

function laplace(dp::DP, logu::Float)
  return laplace(dp.alpha, logu)
end

function laplace(alpha::Float, logu::Float)
  return alpha*log1p(exp(logu))
end

function logGamma(dp::DP, num::Int)
  return logGamma(dp.alpha, num, dp.logU)
end

function logGamma(dp::DP, num::Int, logu::Float)
  return logGamma(dp.alpha, num, logu)
end

function logGamma(alpha::Float, num::Int, logu::Float)
  return log(alpha) - lgamma(1.0) + lgamma(num) - (num)*log(1.+exp(logu))
end

function logMeanMass(dp::DP, num::Int)
  return  logMeanMass(num, dp.logU)
end

function logMeanMass(num::Int, logu::Float)
  return log((num) / (1+exp(logu)))
end

function logMeanTotalMass(dp::DP, numclusters::Int) # why numclusters ??
  return logMeanTotalMass(dp.alpha, numclusters, dp.logU)
end

function logMeanTotalMass(alpha::Float, numclusters::Int, logu::Float) # why numclusters ??
  return log(alpha) + (-1.0)*log(1+exp(logu))
end

function drawLogMass(dp::DP, num::Int, logu::Float)
  return log(rand(Gamma(num,1))) - log(1+exp(logu))
end

function meanNumClusters(dp::DP, num::Int)
  return meanNumClusters(dp, num, dp.logU)
end

function meanNumClusters(dp::DP, num::Int, logu::Float)
  return dp.alpha*log(1+num/dp.alpha)
end

### Sampling
function sample(dp::DP, numData::Int, clusters::Set{Cluster})
  sampleU(dp, numData, clusters)
  sampleAlpha(dp, clusters)
end

function sampleU(dp::DP, numData::Int, clusters::Set{Cluster}) # No prior on hyperparameters
  sampler = SliceStepOut(1.0, 20)

  function logDensityU(logu::Float)
    result = numData *  logu - laplace(dp, logu)
    for cc in clusters
      assert(!isEmpty(cc))
      result += logGamma(dp, cc.number, logu)
    end
    return result
  end

  dp.logU = sample(sampler, logDensityU, dp.logU)
end

function sampleAlpha(dp::DP, clusters::Set{Cluster})
  shape = dp.Ashape + length(clusters)
  invscale = dp.Ainvscale + log1p(exp(dp.logU)) 
  gamma = Gamma(shape, 1.0/invscale)
  dp.alpha = rand(gamma)
end

function logW(dp::DP, s::Float, t::Float)
  return log(dp.alpha) -
  lgamma(1.0) - (1.0)*log(t) - s*(exp(dp.logU)+1.)
end

function invWInt(dp::DP, x::Float, t::Float, logu::Float)
  a = log(x*(exp(logu)+1))
  b = logLevy(dp, t, logu)
  if a >= b  return Inf end
  if b > a+33.0 return t+exp(a-b)/(exp(logu)+1.0)
  else return t-log(1.0-exp(a-b))/(exp(logu)+1.0)
  end
end

function drawLogMasses(dp::DP, slice::Float)
  minSlice = 1.0e-7
  maxClusters = 1000000
  slice = exp(slice)
  if slice < minSlice slice = 1.0e-6 end

  s = slice
  masses = Array{Float}(0)
  while(true)
    e = rand(Exponential(1.0))
    news = invWInt(dp, e, s, dp.logU)
    if news == Inf break end
    if (rand() < exp(logLevy(dp, news, dp.logU) - logW(dp, news, s)))
      push!(masses, news)
    end
    if length(masses) > maxClusters
      masses = Array{Float}(0)
      slice *= 10.0
      news = slice
    end
    s = news
  end

  masses = masses[end:-1:1]
  result = zeros(Float, length(masses))
  for i in 1:length(masses)
    result[i] = log(masses[i])
  end

  return result
end
