import geopandas
import matplotlib.pyplot as plt

file = "Palmyra FAD Watch GIS data for NASA (Nov 2024)-selected\Satlink_FAD_positions_sets_070123_063024_PointsToLine.shp\Satlink_FAD_positions_sets_070123_063024_PointsToLine.shx"
data = geopandas.read_file(file)
splice = data.head(20)

if False:
    plot = splice.plot()

    plt.savefig("test plot.png")

