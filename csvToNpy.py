import sys
import os
import numpy as np

drivers = sorted([int(folderName) for folderName in os.listdir("drivers/")])
# Check that we unzipped the archive properly and have all the drivers
try:
    if len(drivers) != 2736:
        raise
except:
    print "Error: {} drivers found instead of 2736".format(len(drivers))
    sys.exit(0)

trips = range(1, 201)

#drivers = drivers[0:10]
# Main loop
for driver in drivers:
    tripsData = []
    for trip in trips:
        tripsData.append(np.loadtxt("drivers/{}/{}.csv".format(driver, trip), delimiter=',', skiprows=1))
    
    np.save("drivers/{}/trips.npy".format(driver), tripsData)
    print "driver{} done".format(driver)

print "Work done"
