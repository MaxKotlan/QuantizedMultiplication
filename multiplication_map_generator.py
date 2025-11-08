
map_sizes = [4, 8, 16, 32, 64, 128, 256]

for size in map_sizes:
    for x in range(modrange):
        for y in range(modrange):
            q = (x/127) # modified to use 127 instead of 255
            p = (y/127) # modified to use 127 instead of 255
            r = q * p
            g = round(r * 127) + 128 # shift to allow for negative values
            gspace[x,y] = g*(256//size)
