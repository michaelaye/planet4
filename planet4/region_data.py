class Seasoner:
    def get_all_obsids(self):
        return [item for sublist in self.seasons for item in sublist]

    @property
    def all_obsids(self):
        bucket = []
        for i in range(1, 10):
            try:
                attr = getattr(self, f'season{i}')
            except:
                pass
            else:
                bucket += attr
        return bucket

    @property
    def available_seasons(self):
        seasons = []
        for i in range(1, 10):
            s = f"season{i}"
            try:
                attr = getattr(self, s)
            except:
                pass
            else:
                seasons.append(s)
        return seasons

    @property
    def name(self):
        return self.__class__.__name__


class Bilbao(Seasoner):
    lat = -87.008
    lon = 127.273
    season2 = [
        'ESP_011420_0930',
        'ESP_011565_0930',
        'ESP_011776_0930',
        'ESP_012211_0930',
        'ESP_012488_0930',
        'ESP_012633_0930',
        'ESP_012844_0930',

    ]
    season3 = [
        'ESP_021903_0930',
        'ESP_020558_0930',
        'ESP_020347_0930',
    ]


class Oswego_Edge(Seasoner):
    lat = -87
    lon = 86.401
    season2 = [
        'ESP_011408_0930',
        'ESP_011606_0930',
        'ESP_012028_0930',
        'ESP_012463_0930',
        'ESP_012819_0930',
        'ESP_013030_0930',
    ]
    season3 = [
        'ESP_022379_0930',
        'ESP_021654_0930',
        'ESP_021522_0930',
        'ESP_021483_0930',
        'ESP_020955_0930',
        'ESP_020731_0930',
        'ESP_020678_0930',
        'ESP_020533_0930',
        'ESP_020322_0930',
        'ESP_020229_0930',
    ]


class Manhattan_Frontinella(Seasoner):
    lat = -86.987
    lon = 99.367
    season2 = [
        'ESP_011711_0930',
        'ESP_011856_0930',
        'ESP_012634_0930',
        'ESP_012779_0930',
        'ESP_012990_0930',
    ]
    season3 = [
        'ESP_020427_0930',
        'ESP_020282_0930',
        'ESP_020176_0930',
    ]


class Maccelsfield(Seasoner):
    lat = -85.401
    lon = 103.901
    season2 = [
        'ESP_011406_0945',
        'ESP_011407_0945',
        'ESP_011723_0945',
        'ESP_011934_0945',
        'ESP_012079_0945',
        'ESP_012290_0945',
        'ESP_012501_0945',
    ]
    season3 = [
        'ESP_021494_0945',
        'ESP_020927_0945',
        'ESP_020782_0945',
        'ESP_020716_0945',
        'ESP_020571_0945',
        'ESP_020294_0945',
        'ESP_020242_0945',
    ]


class BuenosAires(Seasoner):
    lat = -81.901
    lon = 4.75
    season2 = [
        'ESP_011370_0980',
        'ESP_011515_0980',
        'ESP_011792_0980',
        'ESP_012227_0980',
        'ESP_012504_0980',
        'ESP_012860_0980',
        'ESP_012939_0980',
    ]
    season3 = [
        'ESP_021642_0980',
        'ESP_021497_0980',
        'ESP_020930_0980',
        'ESP_020719_0980',
        'ESP_020508_0980',
        'ESP_020376_0980',
        'ESP_020297_0980',
    ]


class Starburst(Seasoner):
    lat = - 81.801
    lon = 76.14
    season2 = [
        'ESP_011341_0980',
        'ESP_011486_0980',
        'ESP_011697_0980',
        'ESP_011842_0980',
        'ESP_012053_0980',
        'ESP_012264_0980',
        'ESP_012607_0980',
    ]
    season3 = [
        'ESP_021969_0980',
        'ESP_020756_0980',
        'ESP_020677_0980',
    ]


class Potsdam(Seasoner):
    lat = - 81.684,
    lon = 66.28
    season2 = [
        'ESP_011460_0980',
        'ESP_011526_0980',
        'ESP_011737_0980',
        'ESP_012515_0980',
        'ESP_012594_0980',
        'ESP_012805_0980',
        'ESP_012871_0980',
    ]
    season3 = [
        'ESP_022510_0980',
        'ESP_021587_0980',
        'ESP_021574_0980',
        'ESP_021521_0980',
        'ESP_021508_0980',
        'ESP_020941_0980',
        'ESP_020875_0980',
        'ESP_020374_0980',
        'ESP_020163_0980',
    ]


class Portsmouth(Seasoner):
    lat = -87.302
    lon = 167.801
    season2 = [
        'ESP_011960_0925',
        'ESP_012316_0925',
        'ESP_012461_0925',
        'ESP_012527_0925',
        'ESP_012817_0925',
    ]
    season3 = [
        'ESP_021520_0925',
        'ESP_021454_0925',
        'ESP_020953_0925',
        'ESP_020742_0925',
        'ESP_020597_0925',
        'ESP_020386_0925',
    ]


class Manhattan2(Seasoner):
    lat = -85.751
    lon = 105.971
    season1 = ['PSP_002770_0940',
               'PSP_003113_0940']
    season4 = ['ESP_030184_0940',
               'ESP_029762_0940',
               'ESP_029406_0940',
               'ESP_029050_0940',
               'ESP_029024_0940']


class Manhattan(Seasoner):
    lat = -86.39
    lon = 99
    season4 = ['ESP_028932_0935',
               'ESP_028931_0935',
               'ESP_029934_0935',
               'ESP_029657_0935',
               'ESP_029578_0935',
               'ESP_029301_0935',
               'ESP_029090_0935',
               'ESP_029037_0935']
    season3 = ['ESP_022339_0935',
               'ESP_021468_0935',
               'ESP_020888_0935',
               'ESP_020532_0935',
               'ESP_020255_0935',
               'ESP_022260_0935',
               'ESP_021495_0935',
               'ESP_021455_0935',
               'ESP_020954_0935',
               'ESP_020598_0935',
               'ESP_020519_0935',
               'ESP_020321_0935',
               'ESP_020214_0935',
               'ESP_020202_0935']
# Why does Meg have these only in season3:
# ESP_022260_0935
# ESP_021495_0935
# ESP_021455_0935
# ESP_020954_0935
# ESP_020598_0935
# ESP_020519_0935
# ESP_020321_0935
# ESP_020214_0935

    season2 = ['ESP_011394_0935',
               'ESP_011671_0935',
               'ESP_011961_0935',
               'ESP_012251_0935',
               'ESP_012739_0935',
               'ESP_012821_0865',
               'ESP_012884_0935',
               'ESP_013016_0935',
               'ESP_013095_0935']
    season1 = ['PSP_003285_0935',
               'PSP_003430_0935',
               'PSP_003746_0935',
               'PSP_003773_0935',
               'PSP_003997_0935',
               'PSP_004142_0935',
               'PSP_004920_0935',
               'PSP_002532_0935',
               'PSP_002533_0935',
               'PSP_002850_0935',
               'PSP_002876_0935',
               'PSP_002942_0935',
               'PSP_003496_0935',
               'PSP_003523_0935',
               'PSP_003575_0935',
               'PSP_003641_0935',
               'PSP_005579_0935']


class Giza(Seasoner):
    lat = -84.8
    lon = 65.7
    season4 = ['ESP_030106_0950',
               'ESP_029473_0950',
               'ESP_030251_0950',
               'ESP_029328_0950']
    season3 = ['ESP_022273_0950',
               'ESP_021482_0950',
               'ESP_020902_0950',
               'ESP_020783_0950',
               'ESP_020480_0950',
               'ESP_020401_0950',
               'ESP_020150_0950']
    season2 = ['ESP_011447_0950',
               'ESP_011448_0950',
               'ESP_011777_0950',
               'ESP_011843_0950',
               'ESP_012265_0950',
               'ESP_012344_0950',
               'ESP_012753_0950',
               'ESP_012836_0850',
               'ESP_012845_0950',
               'ESP_012212_0950',
               'ESP_012704_0850']
    season1 = ['PSP_004002_0845',
               'PSP_002600_0955',
               'PSP_003246_0950',
               'PSP_003734_0950',
               'PSP_003786_0950',
               'PSP_003787_0950',
               'PSP_003866_0950',
               'PSP_003932_0950',
               'PSP_004024_0950',
               'PSP_004380_0950',
               'PSP_004736_0950',
               'PSP_005119_0950',
               'PSP_003958_0950',
               'PSP_004028_0850',
               'PSP_003474_0850',
               'PSP_004041_0850']


class Inca(Seasoner):
    lat = -81.45
    lon = 296
    season4 = ['ESP_030084_0985',
               'ESP_030229_0985',
               'ESP_029886_0985',
               'ESP_029596_0985',
               'ESP_029240_0985',
               'ESP_029227_0985',
               'ESP_029095_0985',
               'ESP_028911_0985',
               'ESP_028910_0985',
               'ESP_028884_0985',
               'ESP_028752_0985',
               'ESP_030163_0985',
               'ESP_029807_0985',
               'ESP_029741_0985',
               'ESP_029662_0985',
               'ESP_029108_0985']
    season3 = ['ESP_022699_0985',
               'ESP_021460_0985',
               'ESP_020959_0985',
               'ESP_020748_0985',
               'ESP_020194_0985',
               'ESP_020128_0985',
               'ESP_020049_0985',
               'ESP_021829_0985',
               'ESP_021684_0985',
               'ESP_021671_0985',
               'ESP_021605_0985',
               'ESP_021526_0985',
               'ESP_020827_0985',
               'ESP_020339_0985',
               'ESP_020115_0985']
    season2 = ['ESP_013113_0985',
               'ESP_013034_0985',
               'ESP_012889_0985',
               'ESP_012744_0985',
               'ESP_012691_0985',
               'ESP_012467_0985',
               'ESP_012322_0985',
               'ESP_012256_0985',
               'ESP_011900_0985',
               'ESP_011729_0985',
               'ESP_011702_0985',
               'ESP_011623_0985',
               'ESP_011557_0985',
               'ESP_011544_0985',
               'ESP_011491_0985']
    season1 = ['PSP_002380_0985',
               'PSP_002868_0985',
               'PSP_003092_0985',
               'PSP_003158_0985',
               'PSP_003237_0985',
               'PSP_003448_0985',
               'PSP_003593_0985',
               'PSP_003770_0815',
               'PSP_003804_0985',
               'PSP_003928_0815',
               'PSP_004081_0985',
               'PSP_004371_0985']


class Ithaca(Seasoner):
    lat = -85.128
    lon = 180.7
    season3 = ['ESP_021491_0950',
               'ESP_020779_0950',
               'ESP_020568_0950',
               'ESP_020476_0950',
               'ESP_020357_0950',
               'ESP_020146_0950']
    season2 = ['ESP_011350_0945',
               'ESP_011351_0945',
               'ESP_011403_0945',
               'ESP_011404_0945',
               'ESP_011931_0945',
               'ESP_012063_0945',
               'ESP_012076_0945',
               'ESP_012643_0945',
               'ESP_012854_0945',
               'ESP_012858_0855']
    season1 = ['PSP_002622_0945',
               'PSP_002675_0945',
               'PSP_003176_0945',
               'PSP_003193_0850',
               'PSP_003229_0950',
               'PSP_003308_0945',
               'PSP_003309_0945',
               'PSP_003310_0855',
               'PSP_003453_0945',
               'PSP_003466_0945',
               'PSP_003677_0945',
               'PSP_003730_0945',
               'PSP_003756_0945',
               'PSP_003796_0950',
               'PSP_003822_0945',
               'PSP_003954_0945',
               'PSP_004033_0945',
               'PSP_004178_0945',
               'PSP_004666_0945',
               'PSP_004891_0945']


regions = [Giza, Inca, Ithaca, Manhattan, Manhattan2, Bilbao,
           Oswego_Edge, Manhattan_Frontinella, BuenosAires,
           Maccelsfield, Starburst, Potsdam, Portsmouth]


def get_seasons(season):
    result = []
    for region in regions:
        try:
            result.extend(getattr(region, season))
        except AttributeError:
            pass
    return result


season2 = get_seasons('season2')
season3 = get_seasons('season3')
