import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


f1, ax = plt.subplots(1, 1, figsize=(9,8))
m = Basemap(projection='ortho', lat_0=40, lon_0=-70, resolution='l', ax=ax) 

m.drawmapboundary(fill_color=(200/255, 204/255, 208/255), linewidth=0)
m.fillcontinents(color='white', lake_color='lightgray', zorder=0)

# Long/Lat Lines:
# m.drawparallels(np.arange(90,-90,-5), labels=[1,1,1,1], linewidth = 0.25, zorder=1)
# m.drawmeridians(np.arange(-180.,180.,30.), labels=[1,1,1,1], latmax=85, linewidth = 0.25, zorder=1)

# Example data:
# import numpy as np
# size = 1000
# # data to plot
# data = np.arange(size)*0.5/size
# # coordinates
# lat = np.random.uniform(low=65, high=90, size=(size,))
# lon = np.random.uniform(low=-180, high=180, size=(size,))
# x,y = m(lon,lat)
# cmap='viridis'
# m.scatter(x,y,c=data,s=10,cmap=cmap,vmin=0,vmax=0.5,zorder=3,alpha=1)

plt.show()