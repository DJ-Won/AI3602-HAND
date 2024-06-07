import pandas as pd

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__decodemap = {}
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i
del i


def decode_exactly(geohash):
    lat_interval, lon_interval = (21.8853094, 23.4438868), (113.0808116, 115.0277924)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                lon_err /= 2
                if cd & mask:
                    lon_interval = (
                        (lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (
                        lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:
                lat_err /= 2
                if cd & mask:
                    lat_interval = (
                        (lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (
                        lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lon, lat,  lon_err, lat_err


def decode(geohash):
    lon, lat,  _, _ = decode_exactly(geohash)
    return lon, lat


def encode(longitude, latitude, precision=12):
    lat_interval, lon_interval = (21.8853094, 23.4438868), (113.0808116, 115.0277924)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    i = 1
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
        i += 1
    return ''.join(geohash)


def geohash_encode(lon, lat, precision=12):
    '''
    Input latitude and longitude and precision, and encode geohash code

    Parameters
    -------
    lon : Series
        longitude Series
    lat : Series
        latitude Series
    precision : number
        geohash precision

    Returns
    -------
    geohash : Series
        encoded geohash Series
    '''
    tmp = pd.DataFrame()
    tmp['lon'] = lon
    tmp['lat'] = lat
    geohash = tmp.apply(lambda r: encode(
        r['lon'], r['lat'], precision), axis=1)
    return geohash


def geohash_decode(geohash):
    '''
    Decode geohash code

    Parameters
    -------
    geohash : Series
        encoded geohash Series

    Returns
    -------
    lon : Series
        decoded longitude Series
    lat : Series
        decoded latitude Series
    '''
    lonslats = geohash.apply(decode)
    lon = lonslats.apply(lambda r: r[0])
    lat = lonslats.apply(lambda r: r[1])
    return lon, lat

