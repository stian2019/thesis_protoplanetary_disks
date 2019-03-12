# coding: utf-8


import pandas as pd
import glob
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import math
import numpy as np
%matplotlib qt5

Year = 365 * 24 * 60 * 60
2.445974e+13 / Year
# print(glob.glob("PLUTO/cstm/*.dbl"))

AU = 1.49597871e13
GM = 1.3271244e+26

data_dir = ''

data_dir = "PLUTO/cstm/"


out = pd.read_csv(data_dir + "dbl.out", header=None, sep=' ')
out
grid = pd.read_csv(data_dir + "grid.out", header=None, sep=' ',
                   skiprows=9, skipfooter=4, engine='python')
r = (np.array(grid[4]) / 2 + np.array(grid[8]) / 2) / AU
rho = np.fromfile(data_dir + "rho." + str(0).zfill(4) + ".dbl")
#prs = np.fromfile(data_dir + "prs." + str(0).zfill(4) + ".dbl")
vr = np.fromfile(data_dir + "vx1." + str(0).zfill(4) + ".dbl")
vphi = np.fromfile(data_dir + "vx2." + str(0).zfill(4) + ".dbl")
prs = (0.05 * r * AU * vphi / (r * AU))**2. * rho
fig, axs = plt.subplots(2, 2)
# fig.suptitle('adsasasasas')

axs[0][0] = plt.subplot(221)
LineRHO, = axs[0][0].plot(r, rho, 'b-', label=r'$\Sigma_r$')
axs[0][0].loglog()
axs[0][0].legend(loc=0)
axs001 = axs[0][0].twinx()


LineALPHA, = axs001.plot(r, (1 - np.tanh((rho - 15.) / (1.) * 0.2))
                         * 2.e-2 / 2. + 1.e-4, 'g--', label=r'$\alpha$')
axs001.legend()
axs001.loglog()


axs[0][1] = plt.subplot(222)
# axs[0][1].set_title('Presure')
LinePRS, = axs[0][1].plot(r, prs, 'g--', label='$Prs$')
axs[0][1].loglog()
axs[0][1].legend()


axs[1][0] = plt.subplot(223)
# axs[1][0].set_title('Vr')
LineVR, = axs[1][0].plot(r, vr, 'b-', label='$V_r$')
axs[1][0].semilogx()
axs[1][0].legend()

axs[1][1] = plt.subplot(224)
# axs[1][1].set_title('')
LineVPHI, = axs[1][1].plot(r, vphi, 'b-', label=r'$V_{2}$')
LineOmegaK = axs[1][1].plot(
    r, (GM / (r * AU)**3.)**0.5 * r * AU, 'g--', label=r'$V_k=\Omega_{K}*r$')
axs[1][1].legend()
axs[1][1].loglog()


axcolor = 'lightgoldenrodyellow'
axesYear = plt.axes([0., 0., 1., 0.02], facecolor=axcolor)
SliderYear = Slider(axesYear, 'Amp', 1, 100, valinit=0)


def update(val):
    #YearIndex = int(SliderYear.val/100*out[1][len(out[1])-1]/out[1][1])
    for idx in range(len(out[1])):
        if(SliderYear.val / 100 * out[1][len(out[1]) - 1] > out[1][idx]):
            YearIndex = idx
    rho = np.fromfile(data_dir + "rho." + str(YearIndex).zfill(4) + ".dbl")
    #prs = np.fromfile(data_dir + "prs." + str(YearIndex).zfill(4) + ".dbl")
    vr = np.fromfile(data_dir + "vx1." + str(YearIndex).zfill(4) + ".dbl")
    vphi = np.fromfile(data_dir + "vx2." + str(YearIndex).zfill(4) + ".dbl")
    prs = (0.05 * r * AU * vphi / (r * AU))**2. * rho
    LineRHO.set_ydata(rho)
    LineALPHA.set_ydata(
        (1 - np.tanh((rho - 15.) / (1.) * 0.2)) * 2.e-2 / 2. + 1.e-3)
    LinePRS.set_ydata(prs)
    axs[0][0].relim()
    axs[0][0].autoscale_view()
    axs001.relim()
    axs001.autoscale_view()
    axs[0][1].relim()
    axs[0][1].autoscale_view()
    axs[0][1].relim()
    axs[0][1].autoscale_view()
    LineVR.set_ydata(vr)
    axs[1][0].relim()
    axs[1][0].autoscale_view()
    axs[1][1].relim()
    axs[1][1].autoscale_view()
    LineVPHI.set_ydata(vphi)

    fig.suptitle('time: ' + str(out[1][YearIndex] / Year / 1e6)[:5] + ' Myr')


out[1][len(out[1]) - 1] / out[1][1]
out
SliderYear.on_changed(update)
plt.show()


#
#import glob
#import numpy as np
#import matplotlib.pyplot as plt
# Year=365*24*60*60
# 1e10/Year
# print(glob.glob("PLUTO/cstm/*.dbl"))
#import pandas as pd
#out=pd.read_csv("PLUTO/cstm/dbl.out",header=None,sep=' ')
#fig = plt.figure()
# for i in out.index:
#    if i%1==0:
#        #rho=glob.glob():
#        rho=np.fromfile("PLUTO/cstm/rho."+str(i).zfill(4)+".dbl")
#        vr=np.fromfile("PLUTO/cstm/vx1."+str(i).zfill(4)+".dbl")
#        vphi=np.fromfile("PLUTO/cstm/vx2."+str(i).zfill(4)+".dbl")
#        prs=np.fromfile("PLUTO/cstm/prs."+str(i).zfill(4)+".dbl")
#        #d = np.fromfile(i).reshape(4, 300, -1)
#        r = np.logspace(1, np.log10(200), 300)
#        plt.plot(r,rho,label=out[1][i]/Year)
#
# plt.legend()
# plt.loglog()
# plt.show()
# fig.savefig("PLUTO/cstm/tanhalphaHD.png")
