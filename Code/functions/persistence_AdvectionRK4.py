def persistence_AdvectionRK4(particle, fieldset, time):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration.
    with an added persitence model 
    units  WARNING
    particle.dt : (Seconds),
    particle_dlon/dlat : degrees
    particle.age & particle.tau (hours), 
    """
    import numpy as np 
    (u1, v1) = fieldset.UV[particle]
    lon1, lat1 = (particle.lon + u1 * 0.5 * particle.dt, particle.lat + v1 * 0.5 * particle.dt)
    (u2, v2) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat1, lon1, particle]
    lon2, lat2 = (particle.lon + u2 * 0.5 * particle.dt, particle.lat + v2 * 0.5 * particle.dt)
    (u3, v3) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat2, lon2, particle]
    lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
    advection_dlon = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particle.dt  # noqa
    advection_dlat = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particle.dt  # noqa

    ## Calculating persistence 
    persistence_dlon = particle.ui*particle.dt
    persistence_dlat = particle.vi*particle.dt

    # Weighting how much persistence to use
    persistence_frac = np.exp(-particle.age/particle.tau)
    if particle.age < 4*particle.tau: 
        #print(particle.dt, particle.ui, persistence_frac)
        persistence_frac = np.exp(-particle.age/particle.tau)
    else: 
        persistence_frac = 0

    # final displacement 
    particle_dlon += persistence_dlon*persistence_frac + advection_dlon*(1- persistence_frac)
    particle_dlat += persistence_dlat*persistence_frac + advection_dlat*(1- persistence_frac)