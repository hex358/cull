#ifndef ORBIT_SGP4_DEVICE_PROPAGATE_CUH
#define ORBIT_SGP4_DEVICE_PROPAGATE_CUH

// Auto-derived from uploaded vendor/sgp4/sgp4.c propagation section.
// Contains only the propagation-time device functions: deep-space secular,
// deep-space periodic, and the unified SGP4/SDP4 propagate routine.

__device__ static void sgp4__deep_space_secular(
    const sgp4_state_t* state,
    double t,
    double* em, double* argpm, double* inclm,
    double* nodem, double* mm, double* nm,
    double* atime, double* xli, double* xni)
{
    const double step = 720.0;
    const double step2 = step * step / 2.0;

    // Initialize from state
    *atime = state->atime;
    *xli = state->xli;
    *xni = state->xni;

    // Apply lunar-solar periodics
    double zm = state->zmos + 0.017201977 * t;
    double zf = zm + 2.0 * 0.01675 * sin(zm);
    double sinzf = sin(zf);
    double f2 = 0.5 * sinzf * sinzf - 0.25;
    double f3 = -0.5 * sinzf * cos(zf);

    double ses = state->se2 * f2 + state->se3 * f3;
    double sis = state->si2 * f2 + state->si3 * f3;
    double sls = state->sl2 * f2 + state->sl3 * f3 + state->sl4 * sinzf;
    double sghs = state->sgh2 * f2 + state->sgh3 * f3 + state->sgh4 * sinzf;
    double shs = state->sh2 * f2 + state->sh3 * f3;

    zm = state->zmol + 0.22997150 * t;
    zf = zm + 2.0 * 0.05490 * sin(zm);
    sinzf = sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * cos(zf);

    double sel = state->ee2 * f2 + state->e3 * f3;
    double sil = state->xi2 * f2 + state->xi3 * f3;
    double sll = state->xl2 * f2 + state->xl3 * f3 + state->xl4 * sinzf;
    double sghl = state->xgh2 * f2 + state->xgh3 * f3 + state->xgh4 * sinzf;
    double shll = state->xh2 * f2 + state->xh3 * f3;

    double pe = ses + sel;
    double pinc = sis + sil;
    double pl = sls + sll;
    double pgh = sghs + sghl;
    double ph = shs + shll;

    if (fabs(state->inclo) >= 0.2) {
        ph /= sin(state->inclo);
        *inclm += pinc;
        *nodem += ph;
        *argpm -= pgh;
    } else {
        double siniq = sin(state->inclo);
        double cosiq = cos(state->inclo);
        double temp_mod = ph * cosiq;
        *inclm += pinc;
        *nodem += ph / siniq;
        *argpm -= temp_mod / siniq;
    }

    *em += pe;
    *mm += pl;

    // Handle resonance effects
    if (state->irez != 0) {
        const double earthRotRate = 7.29211514668855e-5;

        if (state->irez == 1) {
            // Synchronous resonance terms
            double xfact = state->mdot + state->argpdot + state->nodedot - earthRotRate - state->no_unkozai;
            double xldot = *xni + xfact;
            double theta = fmod(state->gsto + t * earthRotRate, SGP4_TWO_PI);
            double xndt = state->del1 * sin(state->xlamo - 2.0 * (state->nodeo + state->argpo) + theta) +
                   state->del2 * sin(2.0 * (state->xlamo - state->nodeo - state->argpo)) +
                   state->del3 * sin(3.0 * state->xlamo - state->nodeo - state->argpo + theta);

            if (fabs(t - *atime) >= step) {
                double stepp = step;
                if (t < *atime) stepp = -step;

                while (fabs(t - *atime) >= step) {
                    xldot = *xni + xfact;
                    *xli += xldot * stepp + xndt * step2;
                    *xni += xndt * stepp;
                    *atime += stepp;

                    theta = fmod(state->gsto + *atime * earthRotRate, SGP4_TWO_PI);
                    xndt = state->del1 * sin(*xli - 2.0 * (state->nodeo + state->argpo) + theta) +
                           state->del2 * sin(2.0 * (*xli - state->nodeo - state->argpo)) +
                           state->del3 * sin(3.0 * *xli - state->nodeo - state->argpo + theta);
                }
            }

            double ft = t - *atime;
            xldot = *xni + xfact;
            *nm = *xni + xndt * ft;
            double xl = *xli + xldot * ft + xndt * ft * ft * 0.5;
            *mm = xl - 2.0 * *nodem + 2.0 * theta;
        }

        if (state->irez == 2) {
            // Half-day resonance terms
            const double g22 = 5.7686396;
            const double g32 = 0.95240898;
            const double g44 = 1.8014998;
            const double g52 = 1.0508330;
            const double g54 = 4.4108898;

            double xfact = state->mdot + state->dmdt + 2.0 * (state->nodedot + state->dnodt - earthRotRate) - state->no_unkozai;
            double xldot = *xni + xfact;
            double theta = fmod(state->gsto + t * earthRotRate, SGP4_TWO_PI);
            double xomi = state->argpo + state->argpdot * *atime;
            double x2omi = xomi + xomi;
            double x2li = *xli + *xli;

            double xndt = state->d2201 * sin(x2omi + *xli - g22) +
                   state->d2211 * sin(*xli - g22) +
                   state->d3210 * sin(xomi + *xli - g32) +
                   state->d3222 * sin(-xomi + *xli - g32) +
                   state->d4410 * sin(x2omi + x2li - g44) +
                   state->d4422 * sin(x2li - g44) +
                   state->d5220 * sin(xomi + *xli - g52) +
                   state->d5232 * sin(-xomi + *xli - g52) +
                   state->d5421 * sin(xomi + x2li - g54) +
                   state->d5433 * sin(-xomi + x2li - g54);

            double xnddt = state->d2201 * cos(x2omi + *xli - g22) +
                    state->d2211 * cos(*xli - g22) +
                    state->d3210 * cos(xomi + *xli - g32) +
                    state->d3222 * cos(-xomi + *xli - g32) +
                    state->d5220 * cos(xomi + *xli - g52) +
                    state->d5232 * cos(-xomi + *xli - g52) +
                    2.0 * (state->d4410 * cos(x2omi + x2li - g44) +
                           state->d4422 * cos(x2li - g44) +
                           state->d5421 * cos(xomi + x2li - g54) +
                           state->d5433 * cos(-xomi + x2li - g54));
            xnddt *= xldot;

            if (fabs(t - *atime) >= step) {
                double stepp = step;
                if (t < *atime) stepp = -step;

                while (fabs(t - *atime) >= step) {
                    xldot = *xni + xfact;
                    *xli += xldot * stepp + xndt * step2;
                    *xni += xndt * stepp;
                    *atime += stepp;

                    xomi = state->argpo + state->argpdot * *atime;
                    x2omi = xomi + xomi;
                    x2li = *xli + *xli;

                    xndt = state->d2201 * sin(x2omi + *xli - g22) +
                           state->d2211 * sin(*xli - g22) +
                           state->d3210 * sin(xomi + *xli - g32) +
                           state->d3222 * sin(-xomi + *xli - g32) +
                           state->d4410 * sin(x2omi + x2li - g44) +
                           state->d4422 * sin(x2li - g44) +
                           state->d5220 * sin(xomi + *xli - g52) +
                           state->d5232 * sin(-xomi + *xli - g52) +
                           state->d5421 * sin(xomi + x2li - g54) +
                           state->d5433 * sin(-xomi + x2li - g54);

                    xnddt = state->d2201 * cos(x2omi + *xli - g22) +
                            state->d2211 * cos(*xli - g22) +
                            state->d3210 * cos(xomi + *xli - g32) +
                            state->d3222 * cos(-xomi + *xli - g32) +
                            state->d5220 * cos(xomi + *xli - g52) +
                            state->d5232 * cos(-xomi + *xli - g52) +
                            2.0 * (state->d4410 * cos(x2omi + x2li - g44) +
                                   state->d4422 * cos(x2li - g44) +
                                   state->d5421 * cos(xomi + x2li - g54) +
                                   state->d5433 * cos(-xomi + x2li - g54));
                    xnddt *= xldot;
                }
            }

            double ft = t - *atime;
            xldot = *xni + xfact;
            *nm = *xni + xndt * ft + xnddt * ft * ft * 0.5;
            double xl = *xli + xldot * ft + xndt * ft * ft * 0.5;
            double temp_mm = -*nodem - *nodem + theta + theta;
            *mm = xl - xomi + temp_mm;
        }
    }
}

// ============================================================================
// Deep Space Periodic Effects (internal)
// ============================================================================

__device__ static void sgp4__deep_space_periodic(
    const sgp4_state_t* state,
    double t,
    double* em, double* inclm, double* nodem,
    double* argpm, double* mm)
{
    double zm = state->zmos + 0.017201977 * t;
    double zf = zm + 2.0 * 0.01675 * sin(zm);
    double sinzf = sin(zf);
    double f2 = 0.5 * sinzf * sinzf - 0.25;
    double f3 = -0.5 * sinzf * cos(zf);

    double ses = state->se2 * f2 + state->se3 * f3;
    double sis = state->si2 * f2 + state->si3 * f3;
    double sls = state->sl2 * f2 + state->sl3 * f3 + state->sl4 * sinzf;
    double sghs = state->sgh2 * f2 + state->sgh3 * f3 + state->sgh4 * sinzf;
    double shs = state->sh2 * f2 + state->sh3 * f3;

    zm = state->zmol + 0.22997150 * t;
    zf = zm + 2.0 * 0.05490 * sin(zm);
    sinzf = sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * cos(zf);

    double sel = state->ee2 * f2 + state->e3 * f3;
    double sil = state->xi2 * f2 + state->xi3 * f3;
    double sll = state->xl2 * f2 + state->xl3 * f3 + state->xl4 * sinzf;
    double sghl = state->xgh2 * f2 + state->xgh3 * f3 + state->xgh4 * sinzf;
    double shll = state->xh2 * f2 + state->xh3 * f3;

    double pe = ses + sel - state->peo;
    double pinc = sis + sil - state->pinco;
    double pl = sls + sll - state->plo;
    double pgh = sghs + sghl - state->pgho;
    double ph = shs + shll - state->pho;

    if (fabs(state->inclo) >= 0.2) {
        ph /= sin(state->inclo);
        *inclm += pinc;
        *em += pe;
        *nodem += ph;
        *argpm -= pgh;
        *mm += pl;
    } else {
        double siniq = sin(state->inclo);
        double cosiq = cos(state->inclo);

        *inclm += pinc;
        *em += pe;

        double sinis = sin(*inclm);

        if (fabs(*inclm) >= 0.2) {
            double temp_per = ph / sinis;
            *nodem += temp_per;
            *argpm -= pgh - cosiq * temp_per;
            *mm += pl;
        } else {
            double temp_per = ph * cosiq;
            *nodem += ph / siniq;
            *argpm -= temp_per / siniq;
            *mm += pl;
        }
    }
}

// ============================================================================
// SGP4 Propagation
// ============================================================================

__device__ static sgp4_error_t orbit_sgp4_device_propagate(const sgp4_state_t* state, double tsince, sgp4_result_t* result) {
    if (!state || !result) {
        return SGP4_ERROR_NULL_POINTER;
    }

    if (!state->initialized) {
        return SGP4_ERROR_NOT_INITIALIZED;
    }

    const double radiusearthkm = SGP4_RADIUS_EARTH;
    const double xke = SGP4_XKE;
    const double j2 = SGP4_J2;
    const double vkmpersec = SGP4_VKMPERSEC;

    double cosio = cos(state->inclo);
    double sinio = sin(state->inclo);

    // Update for secular gravity and atmospheric drag
    double xmdf = state->mo + state->mdot * tsince;
    double argpdf = state->argpo + state->argpdot * tsince;
    double nodedf = state->nodeo + state->nodedot * tsince;
    double argpm = argpdf;
    double mm = xmdf;
    double t2 = tsince * tsince;
    double nodem = nodedf + state->nodecf * t2;
    double tempa = 1.0 - state->cc1 * tsince;
    double tempe = state->bstar * state->cc4 * tsince;
    double templ = state->t2cof * t2;

    if (!state->isimp) {
        double delomg = state->omgcof * tsince;
        double delm = state->xmcof * (pow(1.0 + state->eta * cos(xmdf), 3) - state->delmo);
        double temp_sgp = delomg + delm;
        mm = xmdf + temp_sgp;
        argpm = argpdf - temp_sgp;
        double t3 = t2 * tsince;
        double t4 = t3 * tsince;
        tempa = tempa - state->d2 * t2 - state->d3 * t3 - state->d4 * t4;
        tempe = tempe + state->bstar * state->cc5 * (sin(mm) - state->sinmao);
        templ = templ + state->t3cof * t3 + t4 * (state->t4cof + tsince * state->t5cof);
    }

    double nm = state->no_unkozai;
    double em = state->ecco;
    double inclm = state->inclo;

    // Initialize resonance state
    double atime = state->atime;
    double xli = state->xli;
    double xni = state->xni;

    // Handle deep space satellites
    if (state->method == 'd') {
        sgp4__deep_space_secular(state, tsince, &em, &argpm, &inclm, &nodem, &mm, &nm, &atime, &xli, &xni);
    }

    double am = pow(xke / nm, SGP4_X2O3) * tempa * tempa;
    nm = xke / pow(am, 1.5);
    em = em - tempe;

    // Check for eccentricity out of range
    if (em >= 1.0 || em < -0.001) {
        return SGP4_ERROR_INVALID_ECCENTRICITY;
    }
    if (em < 1.0e-6) {
        em = 1.0e-6;
    }

    mm = mm + state->no_unkozai * templ;
    double xlm = mm + argpm + nodem;

    nodem = fmod(nodem, SGP4_TWO_PI);
    argpm = fmod(argpm, SGP4_TWO_PI);
    xlm = fmod(xlm, SGP4_TWO_PI);
    mm = fmod(xlm - argpm - nodem, SGP4_TWO_PI);

    // Apply deep space periodic effects
    if (state->method == 'd') {
        sgp4__deep_space_periodic(state, tsince, &em, &inclm, &nodem, &argpm, &mm);
    }

    // Re-compute sini/cosi if inclination changed
    if (inclm != state->inclo) {
        sinio = sin(inclm);
        cosio = cos(inclm);
    }

    if (em < 0.0) {
        em = 1.0e-6;
    }

    double sinim = sin(inclm);
    double cosim = cos(inclm);

    double ep = em;
    double xincp = inclm;
    double argpp = argpm;
    double nodep = nodem;
    double mp = mm;

    // Check if satellite decayed
    if (nm <= 0.0) {
        return SGP4_ERROR_SATELLITE_DECAYED;
    }

    double eccsq = ep * ep;
    double omeosq = 1.0 - eccsq;

    if (omeosq <= 0.0) {
        return SGP4_ERROR_INVALID_ORBIT;
    }

    // Long period periodics
    cosio = cosim;
    sinio = sinim;
    double cosio2 = cosio * cosio;

    double axnl = ep * cos(argpp);
    double temp_lp = 1.0 / (am * omeosq);
    double aynl = ep * sin(argpp) + temp_lp * state->aycof;
    double xl = mp + argpp + nodep + temp_lp * state->xlcof * axnl;

    // Solve Kepler's equation
    double u = fmod(xl - nodep, SGP4_TWO_PI);
    double eo1 = u;
    double tem5 = 9999.9;
    int ktr = 1;
    double sineo1, coseo1;

    while ((fabs(tem5) >= 1.0e-12) && (ktr <= 10)) {
        sineo1 = sin(eo1);
        coseo1 = cos(eo1);
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        if (fabs(tem5) >= 0.95) {
            tem5 = tem5 > 0.0 ? 0.95 : -0.95;
        }
        eo1 = eo1 + tem5;
        ktr++;
    }

    // Short period preliminary quantities
    double ecose = axnl * coseo1 + aynl * sineo1;
    double esine = axnl * sineo1 - aynl * coseo1;
    double el2 = axnl * axnl + aynl * aynl;
    double pl = am * (1.0 - el2);

    if (pl < 0.0) {
        return SGP4_ERROR_INVALID_ORBIT;
    }

    double rl = am * (1.0 - ecose);
    double rdotl = sqrt(am) * esine / rl;
    double rvdotl = sqrt(pl) / rl;
    double betal = sqrt(1.0 - el2);
    double temp_sp = esine / (1.0 + betal);
    double sinu = am / rl * (sineo1 - aynl - axnl * temp_sp);
    double cosu = am / rl * (coseo1 - axnl + aynl * temp_sp);
    double su = atan2(sinu, cosu);
    double sin2u = (cosu + cosu) * sinu;
    double cos2u = 1.0 - 2.0 * sinu * sinu;
    double temp_sp2 = 1.0 / pl;
    double temp1_sp = 0.5 * j2 * temp_sp2;
    double temp2_sp = temp1_sp * temp_sp2;

    // Update for short period periodics
    double con41 = 3.0 * cosio2 - 1.0;
    double x1mth2 = 1.0 - cosio2;
    double x7thm1 = 7.0 * cosio2 - 1.0;

    double mrt = rl * (1.0 - 1.5 * temp2_sp * betal * con41) + 0.5 * temp1_sp * x1mth2 * cos2u;
    su = su - 0.25 * temp2_sp * x7thm1 * sin2u;
    double xnode = nodep + 1.5 * temp2_sp * cosio * sin2u;
    double xinc = xincp + 1.5 * temp2_sp * cosio * sinio * cos2u;
    double mvt = rdotl - nm * temp1_sp * x1mth2 * sin2u / xke;
    double rvdot = rvdotl + nm * temp1_sp * (x1mth2 * cos2u + 1.5 * con41) / xke;

    // Orientation vectors
    double sinsu = sin(su);
    double cossu = cos(su);
    double snod = sin(xnode);
    double cnod = cos(xnode);
    double sini = sin(xinc);
    double cosi = cos(xinc);
    double xmx = -snod * cosi;
    double xmy = cnod * cosi;
    double ux = xmx * sinsu + cnod * cossu;
    double uy = xmy * sinsu + snod * cossu;
    double uz = sini * sinsu;
    double vx = xmx * cossu - cnod * sinsu;
    double vy = xmy * cossu - snod * sinsu;
    double vz = sini * cossu;

    // Position and velocity (in km and km/s)
    result->r[0] = mrt * ux * radiusearthkm;
    result->r[1] = mrt * uy * radiusearthkm;
    result->r[2] = mrt * uz * radiusearthkm;
    result->v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
    result->v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
    result->v[2] = (mvt * uz + rvdot * vz) * vkmpersec;

    // Store updated resonance state
    result->atime = atime;
    result->xli = xli;
    result->xni = xni;

    // Check if satellite decayed
    if (mrt < 1.0) {
        return SGP4_ERROR_SATELLITE_DECAYED;
    }

    return SGP4_SUCCESS;
}



#endif
