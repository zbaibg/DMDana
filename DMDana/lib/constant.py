import numpy as np

#customized constant

#fs  = 41.341373335
#sec = 4.1341373335E+16
#Hatree_to_eV = 27.211386245988
Hatree_to_eV=27.21138505

#Energy, temperature units in Hartrees:
eV = 1/27.21138505 #!< eV in Hartrees
Ryd = 0.5 #!< Rydberg in Hartrees
Joule = 1/4.35974434e-18 #!< Joule in Hartrees
KJoule = 1000*Joule #!< KJoule in Hartrees
Kcal = KJoule * 4.184 #!< Kcal in Hartrees
Kelvin = 1.3806488e-23*Joule #!< Kelvin in Hartrees
invcm = 1./219474.6313705 #!< Inverse cm in Hartrees

#Length units in bohrs:
Angstrom = 1/0.5291772 #!< Angstrom in bohrs
meter = 1e10*Angstrom #!< meter in bohrs
liter = 1e-3*pow(meter,3)  #!< liter in cubic bohrs

#Mass units in electron masses:
amu = 1822.88839 #!< atomic mass unit in electron masses
kg = 1./9.10938291e-31 #!< kilogram in electron masses

#Dimensionless:
mol = 6.0221367e23 #!< mole in number (i.e. Avogadro number)

#Commonly used derived units:
Newton = Joule/meter  #!< Newton in Hartree/bohr
Pascal = Newton/(meter*meter) #!< Pascal in Hartree/bohr^3
KPascal = 1000*Pascal  #!< KPa in Hartree/bohr^3
Bar = 100*KPascal   #!< bar in Hartree/bohr^3
mmHg = 133.322387415*Pascal  #!< mm Hg in Hartree/bohr^3

#Time
sec = np.sqrt((kg*meter)/Newton) #!< second in inverse Hartrees
invSec = 1./sec #!< inverse second in Hartrees
fs = sec*1.0e-15 #!< femtosecond in inverse Hartrees

#Electrical:
Coul = Joule/eV #!< Coulomb in electrons
Volt = Joule/Coul #!< Volt in Hartrees
Ampere = Coul/sec #!< Ampere in electrons/inverse Hartree
Ohm = Volt/Ampere #!< Ohm in inverse conductance quanta

#Magnetic:
Tesla = Volt*sec/(meter*meter) #!< Tesla in atomic units
bohrMagneton = 0.5
gElectron = 2.0023193043617 #!< electron gyromagnetic ratio

#! @}
#endif #JDFTX_CORE_UNITS_H
