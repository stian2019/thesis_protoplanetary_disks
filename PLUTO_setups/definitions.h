#define  PHYSICS                        HD
#define  DIMENSIONS                     1
#define  COMPONENTS                     2
#define  GEOMETRY                       POLAR
#define  BODY_FORCE                     POTENTIAL
#define  FORCED_TURB                    NO
#define  COOLING                        NO
#define  RECONSTRUCTION                 PARABOLIC
#define  TIME_STEPPING                  RK3
#define  DIMENSIONAL_SPLITTING          YES
#define  NTRACER                        0
#define  USER_DEF_PARAMETERS            3

/* -- physics dependent declarations -- */

#define  EOS                            ISOTHERMAL
#define  ENTROPY_SWITCH                 NO
#define  THERMAL_CONDUCTION             NO
#define  VISCOSITY                      SUPER_TIME_STEPPING
#define  ROTATING_FRAME                 NO

/* -- user-defined parameters (labels) -- */

#define  GMstar                         0
#define  AU                             1
#define  GAMMA                          2

/* [Beg] user-defined constants (do not change this line) */

#define  UNIT_DENSITY                   1.
#define  UNIT_LENGTH                    1.
#define  UNIT_VELOCITY                  1.

/* [End] user-defined constants (do not change this line) */
