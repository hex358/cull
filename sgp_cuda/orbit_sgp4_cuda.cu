#include "orbit_sgp4.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "orbit_sgp4_device_propagate.cuh"
#include "vendor/vallado/SGP4.h"

static inline double orbit_sgp4_now_ms(cudaEvent_t start, cudaEvent_t stop) {
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	return (double)ms;
}



#ifndef ORBIT_VALLADO_PI
#define ORBIT_VALLADO_PI 3.141592653589793238462643383279502884
#endif

__device__ static void orbit_vallado_device_dpper
		(
		double e3, double ee2, double peo, double pgho, double pho,
		double pinco, double plo, double se2, double se3, double sgh2,
		double sgh3, double sgh4, double sh2, double sh3, double si2,
		double si3, double sl2, double sl3, double sl4, double t,
		double xgh2, double xgh3, double xgh4, double xh2, double xh3,
		double xi2, double xi3, double xl2, double xl3, double xl4,
		double zmol, double zmos, double inclo,
		char init,
		double& ep, double& inclp, double& nodep, double& argpp, double& mp,
		char opsmode
		)
	{
		/* --------------------- local variables ------------------------ */
		const double twopi = 2.0 * ORBIT_VALLADO_PI;
		double alfdp, betdp, cosip, cosop, dalf, dbet, dls,
			f2, f3, pe, pgh, ph, pinc, pl,
			sel, ses, sghl, sghs, shll, shs, sil,
			sinip, sinop, sinzf, sis, sll, sls, xls,
			xnoh, zf, zm, zel, zes, znl, zns;

		/* ---------------------- constants ----------------------------- */
		zns = 1.19459e-5;
		zes = 0.01675;
		znl = 1.5835218e-4;
		zel = 0.05490;

		/* --------------- calculate time varying periodics ----------- */
		zm = zmos + zns * t;
		// be sure that the initial call has time set to zero
		if (init == 'y')
			zm = zmos;
		zf = zm + 2.0 * zes * sin(zm);
		sinzf = sin(zf);
		f2 = 0.5 * sinzf * sinzf - 0.25;
		f3 = -0.5 * sinzf * cos(zf);
		ses = se2* f2 + se3 * f3;
		sis = si2 * f2 + si3 * f3;
		sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf;
		sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf;
		shs = sh2 * f2 + sh3 * f3;
		zm = zmol + znl * t;
		if (init == 'y')
			zm = zmol;
		zf = zm + 2.0 * zel * sin(zm);
		sinzf = sin(zf);
		f2 = 0.5 * sinzf * sinzf - 0.25;
		f3 = -0.5 * sinzf * cos(zf);
		sel = ee2 * f2 + e3 * f3;
		sil = xi2 * f2 + xi3 * f3;
		sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf;
		sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf;
		shll = xh2 * f2 + xh3 * f3;
		pe = ses + sel;
		pinc = sis + sil;
		pl = sls + sll;
		pgh = sghs + sghl;
		ph = shs + shll;

		if (init == 'n')
		{
			pe = pe - peo;
			pinc = pinc - pinco;
			pl = pl - plo;
			pgh = pgh - pgho;
			ph = ph - pho;
			inclp = inclp + pinc;
			ep = ep + pe;
			sinip = sin(inclp);
			cosip = cos(inclp);

			/* ----------------- apply periodics directly ------------ */
			//  sgp4fix for lyddane choice
			//  strn3 used original inclination - this is technically feasible
			//  gsfc used perturbed inclination - also technically feasible
			//  probably best to readjust the 0.2 limit value and limit discontinuity
			//  0.2 rad = 11.45916 deg
			//  use next line for original strn3 approach and original inclination
			//  if (inclo >= 0.2)
			//  use next line for gsfc version and perturbed inclination
			if (inclp >= 0.2)
			{
				ph = ph / sinip;
				pgh = pgh - cosip * ph;
				argpp = argpp + pgh;
				nodep = nodep + ph;
				mp = mp + pl;
			}
			else
			{
				/* ---- apply periodics with lyddane modification ---- */
				sinop = sin(nodep);
				cosop = cos(nodep);
				alfdp = sinip * sinop;
				betdp = sinip * cosop;
				dalf = ph * cosop + pinc * cosip * sinop;
				dbet = -ph * sinop + pinc * cosip * cosop;
				alfdp = alfdp + dalf;
				betdp = betdp + dbet;
				nodep = fmod(nodep, twopi);
				//  sgp4fix for afspc written intrinsic functions
				// nodep used without a trigonometric function ahead
				if ((nodep < 0.0) && (opsmode == 'a'))
					nodep = nodep + twopi;
				xls = mp + argpp + cosip * nodep;
				dls = pl + pgh - pinc * nodep * sinip;
				xls = xls + dls;
				xnoh = nodep;
				nodep = atan2(alfdp, betdp);
				//  sgp4fix for afspc written intrinsic functions
				// nodep used without a trigonometric function ahead
				if ((nodep < 0.0) && (opsmode == 'a'))
					nodep = nodep + twopi;
				if (fabs(xnoh - nodep) > ORBIT_VALLADO_PI)
					if (nodep < xnoh)
						nodep = nodep + twopi;
					else
						nodep = nodep - twopi;
				mp = mp + pl;
				argpp = xls - mp - cosip * nodep;
			}
		}   // if init == 'n'

		//#include "debug1.cpp"
	}  // dpper

__device__ static void orbit_vallado_device_dspace
		(
		int irez,
		double d2201, double d2211, double d3210, double d3222, double d4410,
		double d4422, double d5220, double d5232, double d5421, double d5433,
		double dedt, double del1, double del2, double del3, double didt,
		double dmdt, double dnodt, double domdt, double argpo, double argpdot,
		double t, double tc, double gsto, double xfact, double xlamo,
		double no,
		double& atime, double& em, double& argpm, double& inclm, double& xli,
		double& mm, double& xni, double& nodem, double& dndt, double& nm
		)
	{
		const double twopi = 2.0 * ORBIT_VALLADO_PI;
		int iretn, iret;
		double delt, ft, theta, x2li, x2omi, xl, xldot, xnddt, xndt, xomi, g22, g32,
			g44, g52, g54, fasx2, fasx4, fasx6, rptim, step2, stepn, stepp;

		fasx2 = 0.13130908;
		fasx4 = 2.8843198;
		fasx6 = 0.37448087;
		g22 = 5.7686396;
		g32 = 0.95240898;
		g44 = 1.8014998;
		g52 = 1.0508330;
		g54 = 4.4108898;
		rptim = 4.37526908801129966e-3; // this equates to 7.29211514668855e-5 rad/sec
		stepp = 720.0;
		stepn = -720.0;
		step2 = 259200.0;

		/* ----------- calculate deep space resonance effects ----------- */
		dndt = 0.0;
		theta = fmod(gsto + tc * rptim, twopi);
		em = em + dedt * t;

		inclm = inclm + didt * t;
		argpm = argpm + domdt * t;
		nodem = nodem + dnodt * t;
		mm = mm + dmdt * t;

		//   sgp4fix for negative inclinations
		//   the following if statement should be commented out
		//  if (inclm < 0.0)
		// {
		//    inclm = -inclm;
		//    argpm = argpm - ORBIT_VALLADO_PI;
		//    nodem = nodem + ORBIT_VALLADO_PI;
		//  }

		/* - update resonances : numerical (euler-maclaurin) integration - */
		/* ------------------------- epoch restart ----------------------  */
		//   sgp4fix for propagator problems
		//   the following integration works for negative time steps and periods
		//   the specific changes are unknown because the original code was so convoluted

		// sgp4fix take out atime = 0.0 and fix for faster operation
		ft = 0.0;
		if (irez != 0)
		{
			// sgp4fix streamline check
			if ((atime == 0.0) || (t * atime <= 0.0) || (fabs(t) < fabs(atime)))
			{
				atime = 0.0;
				xni = no;
				xli = xlamo;
			}
			// sgp4fix move check outside loop
			if (t > 0.0)
				delt = stepp;
			else
				delt = stepn;

			iretn = 381; // added for do loop
			iret = 0; // added for loop
			while (iretn == 381)
			{
				/* ------------------- dot terms calculated ------------- */
				/* ----------- near - synchronous resonance terms ------- */
				if (irez != 2)
				{
					xndt = del1 * sin(xli - fasx2) + del2 * sin(2.0 * (xli - fasx4)) +
						del3 * sin(3.0 * (xli - fasx6));
					xldot = xni + xfact;
					xnddt = del1 * cos(xli - fasx2) +
						2.0 * del2 * cos(2.0 * (xli - fasx4)) +
						3.0 * del3 * cos(3.0 * (xli - fasx6));
					xnddt = xnddt * xldot;
				}
				else
				{
					/* --------- near - half-day resonance terms -------- */
					xomi = argpo + argpdot * atime;
					x2omi = xomi + xomi;
					x2li = xli + xli;
					xndt = d2201 * sin(x2omi + xli - g22) + d2211 * sin(xli - g22) +
						d3210 * sin(xomi + xli - g32) + d3222 * sin(-xomi + xli - g32) +
						d4410 * sin(x2omi + x2li - g44) + d4422 * sin(x2li - g44) +
						d5220 * sin(xomi + xli - g52) + d5232 * sin(-xomi + xli - g52) +
						d5421 * sin(xomi + x2li - g54) + d5433 * sin(-xomi + x2li - g54);
					xldot = xni + xfact;
					xnddt = d2201 * cos(x2omi + xli - g22) + d2211 * cos(xli - g22) +
						d3210 * cos(xomi + xli - g32) + d3222 * cos(-xomi + xli - g32) +
						d5220 * cos(xomi + xli - g52) + d5232 * cos(-xomi + xli - g52) +
						2.0 * (d4410 * cos(x2omi + x2li - g44) +
						d4422 * cos(x2li - g44) + d5421 * cos(xomi + x2li - g54) +
						d5433 * cos(-xomi + x2li - g54));
					xnddt = xnddt * xldot;
				}

				/* ----------------------- integrator ------------------- */
				// sgp4fix move end checks to end of routine
				if (fabs(t - atime) >= stepp)
				{
					iret = 0;
					iretn = 381;
				}
				else // exit here
				{
					ft = t - atime;
					iretn = 0;
				}

				if (iretn == 381)
				{
					xli = xli + xldot * delt + xndt * step2;
					xni = xni + xndt * delt + xnddt * step2;
					atime = atime + delt;
				}
			}  // while iretn = 381

			nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
			xl = xli + xldot * ft + xndt * ft * ft * 0.5;
			if (irez != 1)
			{
				mm = xl - 2.0 * nodem + 2.0 * theta;
				dndt = nm - no;
			}
			else
			{
				mm = xl - nodem - argpm + theta;
				dndt = nm - no;
			}
			nm = no + dndt;
		}

		//#include "debug4.cpp"
	}  // dsspace

__device__ static bool orbit_vallado_device_sgp4
		(
		elsetrec& satrec, double tsince,
		double r[3], double v[3]
		)
	{
		double am, axnl, aynl, betal, cosim, cnod,
			cos2u, coseo1, cosi, cosip, cosisq, cossu, cosu,
			delm, delomg, em, emsq, ecose, el2, eo1,
			ep, esine, argpm, argpp, argpdf, pl, mrt = 0.0,
			mvt, rdotl, rl, rvdot, rvdotl, sinim,
			sin2u, sineo1, sini, sinip, sinsu, sinu,
			snod, su, t2, t3, t4, tem5, temp,
			temp1, temp2, tempa, tempe, templ, u, ux,
			uy, uz, vx, vy, vz, inclm, mm,
			nm, nodem, xinc, xincp, xl, xlm, mp,
			xmdf, xmx, xmy, nodedf, xnode, nodep, tc, dndt,
			twopi, x2o3, vkmpersec, delmtemp;
		int ktr;

		/* ------------------ set mathematical constants --------------- */
		// sgp4fix divisor for divide by zero check on inclination
		// the old check used 1.0 + cos(ORBIT_VALLADO_PI-1.0e-9), but then compared it to
		// 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
		const double temp4 = 1.5e-12;
		twopi = 2.0 * ORBIT_VALLADO_PI;
		x2o3 = 2.0 / 3.0;
		// sgp4fix identify constants and allow alternate values
		// getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
		vkmpersec = satrec.radiusearthkm * satrec.xke / 60.0;

		/* --------------------- clear sgp4 error flag ----------------- */
		satrec.t = tsince;
		satrec.error = 0;

		/* ------- update for secular gravity and atmospheric drag ----- */
		xmdf = satrec.mo + satrec.mdot * satrec.t;
		argpdf = satrec.argpo + satrec.argpdot * satrec.t;
		nodedf = satrec.nodeo + satrec.nodedot * satrec.t;
		argpm = argpdf;
		mm = xmdf;
		t2 = satrec.t * satrec.t;
		nodem = nodedf + satrec.nodecf * t2;
		tempa = 1.0 - satrec.cc1 * satrec.t;
		tempe = satrec.bstar * satrec.cc4 * satrec.t;
		templ = satrec.t2cof * t2;

		if (satrec.isimp != 1)
		{
			delomg = satrec.omgcof * satrec.t;
			// sgp4fix use mutliply for speed instead of pow
			delmtemp = 1.0 + satrec.eta * cos(xmdf);
			delm = satrec.xmcof *
				(delmtemp * delmtemp * delmtemp -
				satrec.delmo);
			temp = delomg + delm;
			mm = xmdf + temp;
			argpm = argpdf - temp;
			t3 = t2 * satrec.t;
			t4 = t3 * satrec.t;
			tempa = tempa - satrec.d2 * t2 - satrec.d3 * t3 -
				satrec.d4 * t4;
			tempe = tempe + satrec.bstar * satrec.cc5 * (sin(mm) -
				satrec.sinmao);
			templ = templ + satrec.t3cof * t3 + t4 * (satrec.t4cof +
				satrec.t * satrec.t5cof);
		}

		nm = satrec.no_unkozai;
		em = satrec.ecco;
		inclm = satrec.inclo;
		if (satrec.method == 'd')
		{
			tc = satrec.t;
			orbit_vallado_device_dspace(
				satrec.irez,
				satrec.d2201, satrec.d2211, satrec.d3210,
				satrec.d3222, satrec.d4410, satrec.d4422,
				satrec.d5220, satrec.d5232, satrec.d5421,
				satrec.d5433, satrec.dedt, satrec.del1,
				satrec.del2, satrec.del3, satrec.didt,
				satrec.dmdt, satrec.dnodt, satrec.domdt,
				satrec.argpo, satrec.argpdot, satrec.t, tc,
				satrec.gsto, satrec.xfact, satrec.xlamo,
				satrec.no_unkozai, satrec.atime,
				em, argpm, inclm, satrec.xli, mm, satrec.xni,
				nodem, dndt, nm
				);
		} // if method = d

		if (nm <= 0.0)
		{
			//         printf("# error nm %f\n", nm);
			satrec.error = 2;
			// sgp4fix add return
			return false;
		}
		am = pow((satrec.xke / nm), x2o3) * tempa * tempa;
		nm = satrec.xke / pow(am, 1.5);
		em = em - tempe;

		// fix tolerance for error recognition
		// sgp4fix am is fixed from the previous nm check
		if ((em >= 1.0) || (em < -0.001)/* || (am < 0.95)*/)
		{
			//         printf("# error em %f\n", em);
			satrec.error = 1;
			// sgp4fix to return if there is an error in eccentricity
			return false;
		}
		// sgp4fix fix tolerance to avoid a divide by zero
		if (em < 1.0e-6)
			em = 1.0e-6;
		mm = mm + satrec.no_unkozai * templ;
		xlm = mm + argpm + nodem;
		emsq = em * em;
		temp = 1.0 - emsq;

		nodem = fmod(nodem, twopi);
		argpm = fmod(argpm, twopi);
		xlm = fmod(xlm, twopi);
		mm = fmod(xlm - argpm - nodem, twopi);

		// sgp4fix recover singly averaged mean elements
		satrec.am = am;
		satrec.em = em;
		satrec.im = inclm;
		satrec.Om = nodem;
		satrec.om = argpm;
		satrec.mm = mm;
		satrec.nm = nm;

		/* ----------------- compute extra mean quantities ------------- */
		sinim = sin(inclm);
		cosim = cos(inclm);

		/* -------------------- add lunar-solar periodics -------------- */
		ep = em;
		xincp = inclm;
		argpp = argpm;
		nodep = nodem;
		mp = mm;
		sinip = sinim;
		cosip = cosim;
		if (satrec.method == 'd')
		{
			orbit_vallado_device_dpper(
				satrec.e3, satrec.ee2, satrec.peo,
				satrec.pgho, satrec.pho, satrec.pinco,
				satrec.plo, satrec.se2, satrec.se3,
				satrec.sgh2, satrec.sgh3, satrec.sgh4,
				satrec.sh2, satrec.sh3, satrec.si2,
				satrec.si3, satrec.sl2, satrec.sl3,
				satrec.sl4, satrec.t, satrec.xgh2,
				satrec.xgh3, satrec.xgh4, satrec.xh2,
				satrec.xh3, satrec.xi2, satrec.xi3,
				satrec.xl2, satrec.xl3, satrec.xl4,
				satrec.zmol, satrec.zmos, satrec.inclo,
				'n', ep, xincp, nodep, argpp, mp, satrec.operationmode
				);
			if (xincp < 0.0)
			{
				xincp = -xincp;
				nodep = nodep + ORBIT_VALLADO_PI;
				argpp = argpp - ORBIT_VALLADO_PI;
			}
			if ((ep < 0.0) || (ep > 1.0))
			{
				//            printf("# error ep %f\n", ep);
				satrec.error = 3;
				// sgp4fix add return
				return false;
			}
		} // if method = d

		/* -------------------- long period periodics ------------------ */
		if (satrec.method == 'd')
		{
			sinip = sin(xincp);
			cosip = cos(xincp);
			satrec.aycof = -0.5*satrec.j3oj2*sinip;
			// sgp4fix for divide by zero for xincp = 180 deg
			if (fabs(cosip + 1.0) > 1.5e-12)
				satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
			else
				satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4;
		}
		axnl = ep * cos(argpp);
		temp = 1.0 / (am * (1.0 - ep * ep));
		aynl = ep* sin(argpp) + temp * satrec.aycof;
		xl = mp + argpp + nodep + temp * satrec.xlcof * axnl;

		/* --------------------- solve kepler's equation --------------- */
		u = fmod(xl - nodep, twopi);
		eo1 = u;
		tem5 = 9999.9;
		ktr = 1;
		//   sgp4fix for kepler iteration
		//   the following iteration needs better limits on corrections
		while ((fabs(tem5) >= 1.0e-12) && (ktr <= 10))
		{
			sineo1 = sin(eo1);
			coseo1 = cos(eo1);
			tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
			tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
			if (fabs(tem5) >= 0.95)
				tem5 = tem5 > 0.0 ? 0.95 : -0.95;
			eo1 = eo1 + tem5;
			ktr = ktr + 1;
		}

		/* ------------- short period preliminary quantities ----------- */
		ecose = axnl*coseo1 + aynl*sineo1;
		esine = axnl*sineo1 - aynl*coseo1;
		el2 = axnl*axnl + aynl*aynl;
		pl = am*(1.0 - el2);
		if (pl < 0.0)
		{
			//         printf("# error pl %f\n", pl);
			satrec.error = 4;
			// sgp4fix add return
			return false;
		}
		else
		{
			rl = am * (1.0 - ecose);
			rdotl = sqrt(am) * esine / rl;
			rvdotl = sqrt(pl) / rl;
			betal = sqrt(1.0 - el2);
			temp = esine / (1.0 + betal);
			sinu = am / rl * (sineo1 - aynl - axnl * temp);
			cosu = am / rl * (coseo1 - axnl + aynl * temp);
			su = atan2(sinu, cosu);
			sin2u = (cosu + cosu) * sinu;
			cos2u = 1.0 - 2.0 * sinu * sinu;
			temp = 1.0 / pl;
			temp1 = 0.5 * satrec.j2 * temp;
			temp2 = temp1 * temp;

			/* -------------- update for short period periodics ------------ */
			if (satrec.method == 'd')
			{
				cosisq = cosip * cosip;
				satrec.con41 = 3.0*cosisq - 1.0;
				satrec.x1mth2 = 1.0 - cosisq;
				satrec.x7thm1 = 7.0*cosisq - 1.0;
			}
			mrt = rl * (1.0 - 1.5 * temp2 * betal * satrec.con41) +
				0.5 * temp1 * satrec.x1mth2 * cos2u;
			su = su - 0.25 * temp2 * satrec.x7thm1 * sin2u;
			xnode = nodep + 1.5 * temp2 * cosip * sin2u;
			xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
			mvt = rdotl - nm * temp1 * satrec.x1mth2 * sin2u / satrec.xke;
			rvdot = rvdotl + nm * temp1 * (satrec.x1mth2 * cos2u +
				1.5 * satrec.con41) / satrec.xke;

			/* --------------------- orientation vectors ------------------- */
			sinsu = sin(su);
			cossu = cos(su);
			snod = sin(xnode);
			cnod = cos(xnode);
			sini = sin(xinc);
			cosi = cos(xinc);
			xmx = -snod * cosi;
			xmy = cnod * cosi;
			ux = xmx * sinsu + cnod * cossu;
			uy = xmy * sinsu + snod * cossu;
			uz = sini * sinsu;
			vx = xmx * cossu - cnod * sinsu;
			vy = xmy * cossu - snod * sinsu;
			vz = sini * cossu;

			/* --------- position and velocity (in km and km/sec) ---------- */
			r[0] = (mrt * ux)* satrec.radiusearthkm;
			r[1] = (mrt * uy)* satrec.radiusearthkm;
			r[2] = (mrt * uz)* satrec.radiusearthkm;
			v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
			v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
			v[2] = (mvt * uz + rvdot * vz) * vkmpersec;
		}  // if pl > 0

		// sgp4fix for decaying satellites
		if (mrt < 1.0)
		{
			//         printf("# decay condition %11.6f \n",mrt);
			satrec.error = 6;
			return false;
		}

		//#include "debug7.cpp"
		return true;
	}  // sgp4


__global__ void orbit_sgp4_vallado_kernel(
	int n_sats,
	int n_times,
	const elsetrec* states,
	const double* jd,
	const double* fr,
	int32_t* errors,
	double* positions,
	double* velocities
) {
	const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = n_sats * n_times;

	if (global_idx >= total) {
		return;
	}

	const int sat_idx = global_idx / n_times;
	const int time_idx = global_idx - sat_idx * n_times;

	const int k1 = global_idx;
	const int k3 = 3 * k1;

	elsetrec satrec = states[sat_idx];

	const double tsince =
		(jd[time_idx] - satrec.jdsatepoch) * 1440.0
		+ (fr[time_idx] - satrec.jdsatepochF) * 1440.0;

	double r[3] = {0.0, 0.0, 0.0};
	double v[3] = {0.0, 0.0, 0.0};

	orbit_vallado_device_sgp4(
		satrec,
		tsince,
		r,
		v
	);

	errors[k1] = (int32_t)satrec.error;

	positions[k3 + 0] = r[0];
	positions[k3 + 1] = r[1];
	positions[k3 + 2] = r[2];

	velocities[k3 + 0] = v[0];
	velocities[k3 + 1] = v[1];
	velocities[k3 + 2] = v[2];

	// Match the existing Vallado CPU bridge behavior:
	// error 6 is a decay condition raised after r/v are produced; errors 1..5
	// are invalid propagation states and must be NaN.
	if (satrec.error && satrec.error < 6) {
		positions[k3 + 0] = NAN;
		positions[k3 + 1] = NAN;
		positions[k3 + 2] = NAN;

		velocities[k3 + 0] = NAN;
		velocities[k3 + 1] = NAN;
		velocities[k3 + 2] = NAN;
	}
}

static int orbit_sgp4_vallado_count_deep_space(
	int n_sats,
	const elsetrec* states
) {
	if (!states || n_sats < 0) {
		return -1;
	}

	int count = 0;

	for (int i = 0; i < n_sats; ++i) {
		if (states[i].method == 'd') {
			++count;
		}
	}

	return count;
}

int orbit_sgp4_vallado_cuda_propagate_states(
	int n_sats,
	const elsetrec* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
) {
	if (n_sats < 0 || n_times < 0 || !states || !jd || !fr || !out) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (!out->errors || !out->positions || !out->velocities) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	const int total_states = n_sats * n_times;
	const int blocks = (total_states + threads_per_block - 1) / threads_per_block;

	const size_t state_bytes = (size_t)n_sats * sizeof(elsetrec);
	const size_t time_bytes = (size_t)n_times * sizeof(double);
	const size_t err_bytes = (size_t)total_states * sizeof(int32_t);
	const size_t vec_bytes = (size_t)total_states * 3u * sizeof(double);

	out->n_sats = n_sats;
	out->n_times = n_times;

	elsetrec* d_states = NULL;
	double* d_jd = NULL;
	double* d_fr = NULL;
	int32_t* d_errors = NULL;
	double* d_positions = NULL;
	double* d_velocities = NULL;

	cudaEvent_t total_start;
	cudaEvent_t h2d_start;
	cudaEvent_t h2d_stop;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	cudaEvent_t d2h_start;
	cudaEvent_t d2h_stop;
	cudaEvent_t total_stop;

	cudaEventCreate(&total_start);
	cudaEventCreate(&h2d_start);
	cudaEventCreate(&h2d_stop);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	cudaEventCreate(&d2h_start);
	cudaEventCreate(&d2h_stop);
	cudaEventCreate(&total_stop);

	cudaError_t ce = cudaSuccess;

	cudaEventRecord(total_start);
	cudaEventRecord(h2d_start);

	ce = cudaMalloc((void**)&d_states, state_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_jd, time_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_fr, time_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_errors, err_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_positions, vec_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_velocities, vec_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_states, states, state_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_jd, jd, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_fr, fr, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(h2d_stop);

	cudaEventRecord(kernel_start);

	orbit_sgp4_vallado_kernel<<<blocks, threads_per_block>>>(
		n_sats,
		n_times,
		d_states,
		d_jd,
		d_fr,
		d_errors,
		d_positions,
		d_velocities
	);

	ce = cudaGetLastError();
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaDeviceSynchronize();
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(kernel_stop);

	cudaEventRecord(d2h_start);

	ce = cudaMemcpy(out->errors, d_errors, err_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(out->positions, d_positions, vec_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(out->velocities, d_velocities, vec_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(d2h_stop);
	cudaEventRecord(total_stop);
	cudaEventSynchronize(total_stop);

	if (stats) {
		memset(stats, 0, sizeof(*stats));

		stats->n_sats = n_sats;
		stats->n_times = n_times;
		stats->state_count = total_states;
		stats->deep_space_count = orbit_sgp4_vallado_count_deep_space(n_sats, states);
		stats->near_earth_count = n_sats - stats->deep_space_count;
		stats->threads_per_block = threads_per_block;
		stats->blocks = blocks;

		stats->h2d_ms = orbit_sgp4_now_ms(h2d_start, h2d_stop);
		stats->kernel_ms = orbit_sgp4_now_ms(kernel_start, kernel_stop);
		stats->d2h_ms = orbit_sgp4_now_ms(d2h_start, d2h_stop);
		stats->total_ms = orbit_sgp4_now_ms(total_start, total_stop);

		int err_count = 0;

		for (int i = 0; i < total_states; ++i) {
			if (out->errors[i] != 0) {
				++err_count;
			}
		}

		stats->error_count = err_count;
	}

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors);
	cudaFree(d_positions);
	cudaFree(d_velocities);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return SGP4_SUCCESS;

cuda_fail:
	fprintf(stderr, "CUDA Vallado failure: %s\n", cudaGetErrorString(ce));

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors);
	cudaFree(d_positions);
	cudaFree(d_velocities);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return -1001;
}


void orbit_sgp4_cuda_free_device_soa(
	orbit_sgp4_device_soa_t* out
) {
	if (!out) {
		return;
	}

	if (out->errors) cudaFree(out->errors);
	if (out->pos_x) cudaFree(out->pos_x);
	if (out->pos_y) cudaFree(out->pos_y);
	if (out->pos_z) cudaFree(out->pos_z);
	if (out->vel_x) cudaFree(out->vel_x);
	if (out->vel_y) cudaFree(out->vel_y);
	if (out->vel_z) cudaFree(out->vel_z);

	memset(out, 0, sizeof(*out));
}


__global__ void orbit_sgp4_aos_to_soa_kernel(
	int n_sats,
	int n_times,
	const int32_t* errors_aos,
	const double* positions_aos,
	const double* velocities_aos,
	int32_t* errors_soa,
	double* pos_x,
	double* pos_y,
	double* pos_z,
	double* vel_x,
	double* vel_y,
	double* vel_z
) {
	const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = n_sats * n_times;

	if (global_idx >= total) {
		return;
	}

	const int sat = global_idx / n_times;
	const int t = global_idx - sat * n_times;

	const int aos_idx = sat * n_times + t;
	const int soa_idx = t * n_sats + sat;

	const int aos3 = 3 * aos_idx;

	errors_soa[soa_idx] = errors_aos[aos_idx];

	pos_x[soa_idx] = positions_aos[aos3 + 0];
	pos_y[soa_idx] = positions_aos[aos3 + 1];
	pos_z[soa_idx] = positions_aos[aos3 + 2];

	if (velocities_aos) {
		vel_x[soa_idx] = velocities_aos[aos3 + 0];
		vel_y[soa_idx] = velocities_aos[aos3 + 1];
		vel_z[soa_idx] = velocities_aos[aos3 + 2];
	}
}


int orbit_sgp4_vallado_cuda_propagate_soa_device(
	int n_sats,
	const void* vallado_states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_device_soa_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
) {
	if (n_sats < 0 || n_times < 0 || !vallado_states || !jd || !fr || !out) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	memset(out, 0, sizeof(*out));

	const elsetrec* states = reinterpret_cast<const elsetrec*>(vallado_states);

	const int total_states = n_sats * n_times;
	const int blocks = (total_states + threads_per_block - 1) / threads_per_block;

	const size_t state_bytes = (size_t)n_sats * sizeof(elsetrec);
	const size_t time_bytes = (size_t)n_times * sizeof(double);
	const size_t err_bytes = (size_t)total_states * sizeof(int32_t);
	const size_t vec_bytes = (size_t)total_states * 3u * sizeof(double);
	const size_t soa_double_bytes = (size_t)total_states * sizeof(double);

	elsetrec* d_states = NULL;
	double* d_jd = NULL;
	double* d_fr = NULL;

	int32_t* d_errors_aos = NULL;
	double* d_positions_aos = NULL;
	double* d_velocities_aos = NULL;

	cudaEvent_t total_start;
	cudaEvent_t h2d_start;
	cudaEvent_t h2d_stop;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	cudaEvent_t d2h_start;
	cudaEvent_t d2h_stop;
	cudaEvent_t total_stop;

	cudaEventCreate(&total_start);
	cudaEventCreate(&h2d_start);
	cudaEventCreate(&h2d_stop);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	cudaEventCreate(&d2h_start);
	cudaEventCreate(&d2h_stop);
	cudaEventCreate(&total_stop);

	cudaError_t ce = cudaSuccess;

	cudaEventRecord(total_start);
	cudaEventRecord(h2d_start);

	ce = cudaMalloc((void**)&d_states, state_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&d_jd, time_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&d_fr, time_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&d_errors_aos, err_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&d_positions_aos, vec_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&d_velocities_aos, vec_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->errors, err_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->pos_x, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->pos_y, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->pos_z, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->vel_x, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->vel_y, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMalloc((void**)&out->vel_z, soa_double_bytes);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMemcpy(d_states, states, state_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMemcpy(d_jd, jd, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaMemcpy(d_fr, fr, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) goto cuda_fail;

	cudaEventRecord(h2d_stop);

	cudaEventRecord(kernel_start);

	orbit_sgp4_vallado_kernel<<<blocks, threads_per_block>>>(
		n_sats,
		n_times,
		d_states,
		d_jd,
		d_fr,
		d_errors_aos,
		d_positions_aos,
		d_velocities_aos
	);

	ce = cudaGetLastError();
	if (ce != cudaSuccess) goto cuda_fail;

	orbit_sgp4_aos_to_soa_kernel<<<blocks, threads_per_block>>>(
		n_sats,
		n_times,
		d_errors_aos,
		d_positions_aos,
		d_velocities_aos,
		out->errors,
		out->pos_x,
		out->pos_y,
		out->pos_z,
		out->vel_x,
		out->vel_y,
		out->vel_z
	);

	ce = cudaGetLastError();
	if (ce != cudaSuccess) goto cuda_fail;

	ce = cudaDeviceSynchronize();
	if (ce != cudaSuccess) goto cuda_fail;

	cudaEventRecord(kernel_stop);

	// Intentional: this device-resident path performs no device-to-host output copy.
	cudaEventRecord(d2h_start);
	cudaEventRecord(d2h_stop);
	cudaEventRecord(total_stop);
	cudaEventSynchronize(total_stop);

	out->n_sats = n_sats;
	out->n_times = n_times;
	out->state_count = total_states;

	if (stats) {
		memset(stats, 0, sizeof(*stats));

		stats->n_sats = n_sats;
		stats->n_times = n_times;
		stats->state_count = total_states;
		stats->deep_space_count = orbit_sgp4_vallado_count_deep_space(n_sats, states);
		stats->near_earth_count = n_sats - stats->deep_space_count;
		stats->threads_per_block = threads_per_block;
		stats->blocks = blocks;
		stats->h2d_ms = orbit_sgp4_now_ms(h2d_start, h2d_stop);
		stats->kernel_ms = orbit_sgp4_now_ms(kernel_start, kernel_stop);
		stats->d2h_ms = 0.0;
		stats->total_ms = orbit_sgp4_now_ms(total_start, total_stop);

		// Keep this zero here. Counting errors would require a D2H copy or a reduction.
		// The downstream LBVH path consumes the device error array directly.
		stats->error_count = 0;
	}

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors_aos);
	cudaFree(d_positions_aos);
	cudaFree(d_velocities_aos);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return SGP4_SUCCESS;

cuda_fail:
	fprintf(stderr, "CUDA Vallado device-SOA failure: %s\n", cudaGetErrorString(ce));

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors_aos);
	cudaFree(d_positions_aos);
	cudaFree(d_velocities_aos);

	orbit_sgp4_cuda_free_device_soa(out);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return -1002;
}


__global__ void orbit_sgp4_kernel(
	int n_sats,
	int n_times,
	const sgp4_state_t* states,
	const double* jd,
	const double* fr,
	int32_t* errors,
	double* positions,
	double* velocities
) {
	const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = n_sats * n_times;

	if (global_idx >= total) {
		return;
	}

	const int sat_idx = global_idx / n_times;
	const int time_idx = global_idx - sat_idx * n_times;

	const int k1 = global_idx;
	const int k3 = 3 * k1;

	const sgp4_state_t* state = &states[sat_idx];

	const double tsince =
		(jd[time_idx] - state->jdsatepoch) * 1440.0
		+ (fr[time_idx] - state->jdsatepochF) * 1440.0;

	sgp4_result_t result;
	result.r[0] = result.r[1] = result.r[2] = 0.0;
	result.v[0] = result.v[1] = result.v[2] = 0.0;
	result.atime = 0.0;
	result.xli = 0.0;
	result.xni = 0.0;

	const sgp4_error_t err = orbit_sgp4_device_propagate(
		state,
		tsince,
		&result
	);

	errors[k1] = (int32_t)err;

	if (err == SGP4_SUCCESS) {
		positions[k3 + 0] = result.r[0];
		positions[k3 + 1] = result.r[1];
		positions[k3 + 2] = result.r[2];

		velocities[k3 + 0] = result.v[0];
		velocities[k3 + 1] = result.v[1];
		velocities[k3 + 2] = result.v[2];
	} else {
		positions[k3 + 0] = NAN;
		positions[k3 + 1] = NAN;
		positions[k3 + 2] = NAN;

		velocities[k3 + 0] = NAN;
		velocities[k3 + 1] = NAN;
		velocities[k3 + 2] = NAN;
	}
}

int orbit_sgp4_cuda_propagate_states(
	int n_sats,
	const sgp4_state_t* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
) {
	if (n_sats < 0 || n_times < 0 || !states || !jd || !fr || !out) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (!out->errors || !out->positions || !out->velocities) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	const int total_states = n_sats * n_times;
	const int blocks = (total_states + threads_per_block - 1) / threads_per_block;

	const size_t state_bytes = (size_t)n_sats * sizeof(sgp4_state_t);
	const size_t time_bytes = (size_t)n_times * sizeof(double);
	const size_t err_bytes = (size_t)total_states * sizeof(int32_t);
	const size_t vec_bytes = (size_t)total_states * 3u * sizeof(double);

	out->n_sats = n_sats;
	out->n_times = n_times;

	sgp4_state_t* d_states = NULL;
	double* d_jd = NULL;
	double* d_fr = NULL;
	int32_t* d_errors = NULL;
	double* d_positions = NULL;
	double* d_velocities = NULL;

	cudaEvent_t total_start;
	cudaEvent_t h2d_start;
	cudaEvent_t h2d_stop;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	cudaEvent_t d2h_start;
	cudaEvent_t d2h_stop;
	cudaEvent_t total_stop;

	cudaEventCreate(&total_start);
	cudaEventCreate(&h2d_start);
	cudaEventCreate(&h2d_stop);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	cudaEventCreate(&d2h_start);
	cudaEventCreate(&d2h_stop);
	cudaEventCreate(&total_stop);

	cudaError_t ce = cudaSuccess;

	cudaEventRecord(total_start);
	cudaEventRecord(h2d_start);

	ce = cudaMalloc((void**)&d_states, state_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_jd, time_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_fr, time_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_errors, err_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_positions, vec_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMalloc((void**)&d_velocities, vec_bytes);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_states, states, state_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_jd, jd, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(d_fr, fr, time_bytes, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(h2d_stop);

	cudaEventRecord(kernel_start);

	orbit_sgp4_kernel<<<blocks, threads_per_block>>>(
		n_sats,
		n_times,
		d_states,
		d_jd,
		d_fr,
		d_errors,
		d_positions,
		d_velocities
	);

	ce = cudaGetLastError();
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaDeviceSynchronize();
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(kernel_stop);

	cudaEventRecord(d2h_start);

	ce = cudaMemcpy(out->errors, d_errors, err_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(out->positions, d_positions, vec_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	ce = cudaMemcpy(out->velocities, d_velocities, vec_bytes, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		goto cuda_fail;
	}

	cudaEventRecord(d2h_stop);
	cudaEventRecord(total_stop);
	cudaEventSynchronize(total_stop);

	if (stats) {
		memset(stats, 0, sizeof(*stats));

		stats->n_sats = n_sats;
		stats->n_times = n_times;
		stats->state_count = total_states;
		stats->deep_space_count = orbit_sgp4_count_deep_space(n_sats, states);
		stats->near_earth_count = n_sats - stats->deep_space_count;
		stats->threads_per_block = threads_per_block;
		stats->blocks = blocks;

		stats->h2d_ms = orbit_sgp4_now_ms(h2d_start, h2d_stop);
		stats->kernel_ms = orbit_sgp4_now_ms(kernel_start, kernel_stop);
		stats->d2h_ms = orbit_sgp4_now_ms(d2h_start, d2h_stop);
		stats->total_ms = orbit_sgp4_now_ms(total_start, total_stop);

		int err_count = 0;

		for (int i = 0; i < total_states; ++i) {
			if (out->errors[i] != SGP4_SUCCESS) {
				++err_count;
			}
		}

		stats->error_count = err_count;
	}

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors);
	cudaFree(d_positions);
	cudaFree(d_velocities);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return SGP4_SUCCESS;

cuda_fail:
	fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(ce));

	cudaFree(d_states);
	cudaFree(d_jd);
	cudaFree(d_fr);
	cudaFree(d_errors);
	cudaFree(d_positions);
	cudaFree(d_velocities);

	cudaEventDestroy(total_start);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);
	cudaEventDestroy(total_stop);

	return -1000;
}