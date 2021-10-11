
import pandas as pd

from typing import Any, Dict, List, Tuple, Union


class Metadata:
    """Define the metadata of the supported datasets and
    a few class methods that are useful to extract info from the metadata of a dataset.

    Attributes
    ----------
    DATASETS : Dict[str, Union[Dict[str, Any], Dict[int, Any]]]
        The metadata of the supported datasets.
    """

    DATASETS: Dict[str, Union[Dict[str, Any], Dict[int, Any]]] = {
        'adult': {  # https://archive.ics.uci.edu/ml/datasets/Adult
            'age': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 17, 'max': 90, 'mean': 38.58164675532078, 'median': 37.0,
                    'std': 13.640432553581146, 'skewness': 0.5587433694130484, 'kurtosis': -0.16612745957143904
                }
            },
            'workclass': {  # the UCI dataset does NOT have miss. values but the CTGAN has, thus, the list is NOT empty
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'Private': 22696, 'Self-emp-not-inc': 2541, 'Local-gov': 2093, 'State-gov': 1298,
                    'Self-emp-inc': 1116, 'Federal-gov': 960, 'Without-pay': 14, 'Never-worked': 7, '?': 1836
                }
            },
            'fnlwgt': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 12285, 'max': 1484705, 'mean': 189778.36651208502, 'median': 178356.0,
                    'std': 105549.97769702233, 'skewness': 1.4469800945789826, 'kurtosis': 6.218810978153801
                }
            },
            'education': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'HS-grad': 10501, 'Some-college': 7291, 'Bachelors': 5355, 'Masters': 1723, 'Assoc-voc': 1382,
                    '11th': 1175, 'Assoc-acdm': 1067, '10th': 933, '7th-8th': 646, 'Prof-school': 576, '9th': 514,
                    '12th': 433, 'Doctorate': 413, '5th-6th': 333, '1st-4th': 168, 'Preschool': 51
                }
            },
            'education-num': {  # this is an alias of the `education` variable (i.e., feature/column)
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    1: 51, 2: 168, 3: 333, 4: 646, 5: 514, 6: 933, 7: 1175, 8: 433,
                    9: 10501, 10: 7291, 11: 1382, 12: 1067, 13: 5355, 14: 1723, 15: 576, 16: 413
                    # 'min': 1, 'max': 16, 'mean': 10.0806793403151, 'median': 10.0,
                    # 'std': 2.5727203320673406, 'skewness': -0.3116758679102297, 'kurtosis': 0.6234440747629248
                }
            },
            'marital-status': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Married-civ-spouse': 14976, 'Never-married': 10683, 'Divorced': 4443, 'Separated': 1025,
                    'Widowed': 993, 'Married-spouse-absent': 418, 'Married-AF-spouse': 23
                }
            },
            'occupation': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'Prof-specialty': 4140, 'Craft-repair': 4099, 'Exec-managerial': 4066, 'Adm-clerical': 3770,
                    'Sales': 3650, 'Other-service': 3295, 'Machine-op-inspct': 2002, 'Transport-moving': 1597,
                    'Handlers-cleaners': 1370, 'Farming-fishing': 994, 'Tech-support': 928, 'Protective-serv': 649,
                    'Priv-house-serv': 149, 'Armed-Forces': 9, '?': 1843
                }
            },
            'relationship': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Husband': 13193, 'Not-in-family': 8305, 'Own-child': 5068,
                    'Unmarried': 3446, 'Wife': 1568, 'Other-relative': 981
                }
            },
            'race': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'White': 27816, 'Black': 3124, 'Asian-Pac-Islander': 1039, 'Amer-Indian-Eskimo': 311, 'Other': 271
                }
            },
            'sex': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Male': 21790, 'Female': 10771
                }
            },
            'capital-gain': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 99999, 'mean': 1077.6488437087312, 'median': 0.0,
                    'std': 7385.292084839299, 'skewness': 11.953847687699799, 'kurtosis': 154.79943785425334
                }
            },
            'capital-loss': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 4356, 'mean': 87.303829734959, 'median': 0.0,
                    'std': 402.960218649059, 'skewness': 4.594629121679692, 'kurtosis': 20.3768017134122
                }
            },
            'hours-per-week': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1, 'max': 99, 'mean': 40.437455852092995, 'median': 40.0,
                    'std': 12.34742868173081, 'skewness': 0.22764253680450092, 'kurtosis': 2.916686796002066
                }
            },
            'native-country': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'United-States': 29170, 'Mexico': 643, 'Philippines': 198, 'Germany': 137, 'Canada': 121,
                    'Puerto-Rico': 114, 'El-Salvador': 106, 'India': 100, 'Cuba': 95, 'England': 90, 'Jamaica': 81,
                    'South': 80, 'China': 75, 'Italy': 73, 'Dominican-Republic': 70, 'Vietnam': 67, 'Guatemala': 64,
                    'Japan': 62, 'Poland': 60, 'Columbia': 59, 'Taiwan': 51, 'Haiti': 44, 'Iran': 43, 'Portugal': 37,
                    'Nicaragua': 34, 'Peru': 31, 'France': 29, 'Greece': 29, 'Ecuador': 28, 'Ireland': 24, 'Hong': 20,
                    'Cambodia': 19, 'Trinadad&Tobago': 19, 'Laos': 18, 'Thailand': 18, 'Yugoslavia': 16, 'Hungary': 13,
                    'Outlying-US(Guam-USVI-etc)': 14, 'Honduras': 13, 'Scotland': 12, 'Holand-Netherlands': 1, '?': 583
                }
            },
            'income': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    '<=50K': 24720, '>50K': 7841
                }
            }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 10 %
            #     'age': {
            #         'min': 30, 'max': 49, 'mean': 38.2094688681122, 'median': 37.0,
            #         'std': 4.652923719305053, 'skewness': 0.2923593624139284, 'kurtosis': -1.2578045208642308
            #     },
            #     'workclass': {'Private': 30162},
            #     'fnlwgt': {
            #         'min': 152749, 'max': 252925, 'mean': 191143.1789006034, 'median': 190293.5,
            #         'std': 12781.38572385394, 'skewness': 0.3738202997696625, 'kurtosis': 0.15807751961954564
            #     },
            #     'education': {'10th': 30162},
            #     'marital-status': {'Married-civ-spouse': 13680, 'Never-married': 9275, 'Divorced': 7207},
            #     'occupation': {'Adm-clerical': 30162},
            #     'relationship': {'Husband': 25169, 'Not-in-family': 4993},
            #     'race': {'White': 30162},
            #     'sex': {'Male': 18479, 'Female': 11683},
            #     'capital-gain': {
            #         'min': 687, 'max': 2655, 'mean': 1280.9614415489689, 'median': 1253.0,
            #         'std': 233.78243246049954, 'skewness': 0.7168980223973471, 'kurtosis': 0.7886063435005122
            #     },
            #     'capital-loss': {
            #         'min': 49, 'max': 181, 'mean': 87.961176314568, 'median': 86.0,
            #         'std': 14.292050519395406, 'skewness': 0.6693303707332083, 'kurtosis': 0.768002412060464
            #     },
            #     'hours-per-week': {
            #         'min': 33, 'max': 49, 'mean': 40.770704860420395, 'median': 40.0,
            #         'std': 3.885654670484348, 'skewness': 0.16409971320115493, 'kurtosis': -1.2899924873230593
            #     },
            #     'native-country': {'United-States': 30162},
            #     'income': {'<=50K': 24972, '>50K': 5190}
            # }
            ############################################################################################################
            # {  # SynSGAIN-CP --> missing rate = 10 %
            #     'age': {
            #         'min': 36, 'max': 41, 'mean': 37.9596180624627, 'median': 38.0,
            #         'std': 0.6477464306985328, 'skewness': 0.12213210059437125, 'kurtosis': -0.003308226753074184
            #     },
            #     'workclass': {'Private': 30162},
            #     'fnlwgt': {
            #         'min': 155158, 'max': 246749, 'mean': 192490.80581526423, 'median': 191871.0,
            #         'std': 11660.288941016035, 'skewness': 0.3461587251000595, 'kurtosis': 0.15249829082735733
            #     },
            #     'education': {'10th': 30162},
            #     'marital-status': {'Divorced': 24303, 'Married-civ-spouse': 5859},
            #     'occupation': {'Adm-clerical': 30162},
            #     'relationship': {'Husband': 30162},
            #     'race': {'White': 30162},
            #     'sex': {'Male': 30162},
            #     'capital-gain': {
            #         'min': 1050, 'max': 3628, 'mean': 1791.036436575824, 'median': 1763.0,
            #         'std': 271.0773993389212, 'skewness': 0.6859910373875169, 'kurtosis': 0.8952455745938717
            #     },
            #     'capital-loss': {
            #         'min': 61, 'max': 190, 'mean': 99.21358000132618, 'median': 98.0,
            #         'std': 13.878316758869849, 'skewness': 0.6303170124943663, 'kurtosis': 0.7332399549342634
            #     },
            #     'hours-per-week': {
            #         'min': 38, 'max': 43, 'mean': 40.74411511172999, 'median': 41.0,
            #         'std': 0.6602749525071553, 'skewness': -0.4304237671246357, 'kurtosis': 0.3734576965036216
            #     },
            #     'native-country': {'United-States': 30162},
            #     'income': {'<=50K': 30162}
            # }
            ############################################################################################################
            # {  # SynSGAIN-GP --> missing rate = 10 %
            #     'age': {
            #         'min': 36, 'max': 40, 'mean': 37.87507459717526, 'median': 38.0,
            #         'std': 0.6975895037562039, 'skewness': 0.08531912838436201, 'kurtosis': -0.07538771535450461
            #     },
            #     'workclass': {'Private': 30162},
            #     'fnlwgt': {
            #         'min': 154974, 'max': 248618, 'mean': 191338.3275313308, 'median': 190555.5,
            #         'std': 11585.005297599584, 'skewness': 0.3864431201560824, 'kurtosis': 0.21572262266951947
            #     },
            #     'education': {'10th': 30162},
            #     'marital-status': {'Divorced': 26553, 'Married-civ-spouse': 3609},
            #     'occupation': {'Adm-clerical': 30162},
            #     'relationship': {'Husband': 30162},
            #     'race': {'White': 30162},
            #     'sex': {'Male': 30162},
            #     'capital-gain': {
            #         'min': 1040, 'max': 3430, 'mean': 1782.9769909157217, 'median': 1755.0,
            #         'std': 266.70728244871776, 'skewness': 0.64409407620106, 'kurtosis': 0.6853266998216601
            #     },
            #     'capital-loss': {
            #         'min': 62, 'max': 181, 'mean': 97.73748425170744, 'median': 96.0,
            #         'std': 13.818737570788223, 'skewness': 0.6222788478997535, 'kurtosis': 0.6434270506919297
            #     },
            #     'hours-per-week': {
            #         'min': 39, 'max': 42, 'mean': 40.49585571248591, 'median': 40.0,
            #         'std': 0.5519340642697685, 'skewness': 0.14533832540510908, 'kurtosis': -0.8503490877328819
            #     },
            #     'native-country': {'United-States': 30162},
            #     'income': {'<=50K': 30162}
            # }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 50 %
            #     'age': {
            #         'min': 31, 'max': 47, 'mean': 38.29656521450832, 'median': 38.0,
            #         'std': 2.2203789529595377, 'skewness': 0.10485820673348566, 'kurtosis': -0.4667801383211314
            #     },
            #     'workclass': {'Private': 30162},
            #     'fnlwgt': {
            #         'min': 93526, 'max': 351408, 'mean': 195982.762482594, 'median': 193816.0,
            #         'std': 29782.23305785774, 'skewness': 0.4209067882025326, 'kurtosis': 0.2516108785462041
            #     },
            #     'education': {'10th': 30162},
            #     'marital-status': {'Married-civ-spouse': 13895, 'Divorced': 10612, 'Never-married': 5655},
            #     'occupation': {'Adm-clerical': 30162},
            #     'relationship': {'Husband': 29772, 'Not-in-family': 390},
            #     'race': {'White': 30162},
            #     'sex': {'Male': 22739, 'Female': 7423},
            #     'capital-gain': {
            #         'min': 203, 'max': 7018, 'mean': 1543.6664345865659, 'median': 1422.0,
            #         'std': 650.46181386814, 'skewness': 1.2851804390862942, 'kurtosis': 2.9732123890388458
            #     },
            #     'capital-loss': {
            #         'min': 19, 'max': 445, 'mean': 92.43352562827398, 'median': 87.0,
            #         'std': 35.03601391579648, 'skewness': 1.1133112314303542, 'kurtosis': 2.402695999280194
            #     },
            #     'hours-per-week': {
            #         'min': 34, 'max': 48, 'mean': 40.69375372985876, 'median': 41.0,
            #         'std': 2.072603818564764, 'skewness': 0.09029397018791628, 'kurtosis': -0.5540347857537724
            #     },
            #     'native-country': {'United-States': 30162},
            #     'income': {'<=50K': 29921, '>50K': 241}
            # }
            ############################################################################################################
        },       # ok
        'breast': {  # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
            'ID': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    883263: 1, 906564: 1, 89122: 1, 9013579: 1, 868682: 1, 859465: 1, 859464: 1, 911685: 1, 895299: 1,
                    909220: 1, 8811842: 1, 916799: 1, 901034302: 1, 901034301: 1, 8911164: 1, 869691: 1, 8812877: 1,
                    859471: 1, 911673: 1, 87281702: 1, 85638502: 1, 91762702: 1, 859487: 1, 857438: 1, 91485: 1,
                    903516: 1, 9013594: 1, 914769: 1, 8611161: 1, 9012568: 1, 874839: 1, 905557: 1, 862548: 1, 86355: 1,
                    903483: 1, 88199202: 1, 90317302: 1, 8811779: 1, 91805: 1, 8812818: 1, 8812816: 1, 86973702: 1,
                    86973701: 1, 904969: 1, 84358402: 1, 891703: 1, 89344: 1, 857343: 1, 9111805: 1, 854268: 1,
                    88119002: 1, 873885: 1, 862485: 1, 927241: 1, 892214: 1, 906539: 1, 881972: 1, 879804: 1,
                    89143602: 1, 89143601: 1, 857392: 1, 8812844: 1, 9047: 1, 857373: 1, 8710441: 1, 9011495: 1,
                    9011494: 1, 924964: 1, 9111843: 1, 857374: 1, 8510824: 1, 874858: 1, 924342: 1, 909445: 1,
                    888264: 1, 904647: 1, 90524101: 1, 901315: 1, 91544002: 1, 91544001: 1, 859575: 1, 8810955: 1,
                    911320502: 1, 911320501: 1, 89524: 1, 88249602: 1, 914101: 1, 91227: 1, 85382601: 1, 861648: 1,
                    858477: 1, 8670: 1, 904689: 1, 892399: 1, 8810987: 1, 887181: 1, 915940: 1, 9010018: 1, 926682: 1,
                    89742801: 1, 868826: 1, 864729: 1, 864726: 1, 884180: 1, 91903901: 1, 886226: 1, 864685: 1,
                    8910251: 1, 905978: 1, 894329: 1, 912193: 1, 909777: 1, 903554: 1, 905190: 1, 8610175: 1,
                    8911230: 1, 906616: 1, 9010598: 1, 8912909: 1, 898678: 1, 873843: 1, 866674: 1, 91505: 1, 91504: 1,
                    86408: 1, 86409: 1, 84348301: 1, 922840: 1, 846226: 1, 9010333: 1, 89869: 1, 8912280: 1, 853401: 1,
                    866714: 1, 867739: 1, 8811523: 1, 861597: 1, 861598: 1, 9110944: 1, 875938: 1, 877989: 1,
                    9113846: 1, 8612080: 1, 8912284: 1, 90944601: 1, 8712289: 1, 894047: 1, 855133: 1, 8610908: 1,
                    9012315: 1, 9012795: 1, 87127: 1, 8712291: 1, 909231: 1, 9010259: 1, 9010258: 1, 899147: 1,
                    87139402: 1, 857156: 1, 855138: 1, 869476: 1, 87163: 1, 897137: 1, 873592: 1, 913102: 1, 849014: 1,
                    905539: 1, 899187: 1, 873586: 1, 907367: 1, 913505: 1, 904302: 1, 897132: 1, 87556202: 1, 901011: 1,
                    913512: 1, 84300903: 1, 857155: 1, 87106: 1, 91376702: 1, 891923: 1, 8810528: 1, 911391: 1,
                    8860702: 1, 911384: 1, 854039: 1, 90439701: 1, 9112594: 1, 91376701: 1, 908445: 1, 8911670: 1,
                    843786: 1, 84667401: 1, 911366: 1, 915460: 1, 8711202: 1, 864292: 1, 863270: 1, 924934: 1,
                    856106: 1, 9111596: 1, 8610862: 1, 8711216: 1, 84862001: 1, 906290: 1, 912600: 1, 862261: 1,
                    911157302: 1, 902727: 1, 897374: 1, 867387: 1, 859196: 1, 873593: 1, 87164: 1, 923169: 1, 86211: 1,
                    878796: 1, 922576: 1, 908489: 1, 90312: 1, 893548: 1, 881861: 1, 90769602: 1, 89296: 1, 90769601: 1,
                    86208: 1, 923748: 1, 8510653: 1, 865468: 1, 90401602: 1, 8810703: 1, 853201: 1, 855167: 1,
                    91594602: 1, 854253: 1, 915691: 1, 923465: 1, 891670: 1, 905501: 1, 873701: 1, 871001502: 1,
                    884948: 1, 88649001: 1, 871642: 1, 871641: 1, 9113816: 1, 905520: 1, 879830: 1, 88995002: 1,
                    8912055: 1, 88299702: 1, 883852: 1, 859283: 1, 881094802: 1, 907409: 1, 865423: 1, 871201: 1,
                    891716: 1, 90251: 1, 844981: 1, 894090: 1, 894089: 1, 9112712: 1, 912519: 1, 893061: 1, 923780: 1,
                    8910996: 1, 8915: 1, 865432: 1, 8913049: 1, 866458: 1, 871122: 1, 88147102: 1, 872608: 1, 909411: 1,
                    904357: 1, 874662: 1, 901288: 1, 912558: 1, 8912049: 1, 9113778: 1, 90291: 1, 916221: 1, 86517: 1,
                    871001501: 1, 8910988: 1, 845636: 1, 862028: 1, 921386: 1, 87880: 1, 896839: 1, 91813702: 1,
                    8610629: 1, 898690: 1, 924632: 1, 89864002: 1, 90401601: 1, 902976: 1, 902975: 1, 90602302: 1,
                    911654: 1, 8610637: 1, 859983: 1, 862009: 1, 896864: 1, 865128: 1, 901303: 1, 858981: 1, 903011: 1,
                    869218: 1, 911201: 1, 911202: 1, 917092: 1, 8711003: 1, 858970: 1, 9110720: 1, 897880: 1, 893783: 1,
                    883539: 1, 8911163: 1, 882488: 1, 9013838: 1, 862980: 1, 842517: 1, 864018: 1, 862989: 1, 904971: 1,
                    888570: 1, 86730502: 1, 9011971: 1, 8953902: 1, 917897: 1, 875263: 1, 887549: 1, 907145: 1,
                    88203002: 1, 862965: 1, 88206102: 1, 925292: 1, 863031: 1, 921385: 1, 863030: 1, 8911834: 1,
                    9112367: 1, 911150: 1, 852781: 1, 88518501: 1, 906024: 1, 852763: 1, 85713702: 1, 866083: 1,
                    91930402: 1, 864033: 1, 9012000: 1, 91858: 1, 858986: 1, 895100: 1, 9113239: 1, 914366: 1,
                    884689: 1, 8611792: 1, 9110732: 1, 8810436: 1, 9113538: 1, 918465: 1, 877501: 1, 915276: 1,
                    877500: 1, 89813: 1, 8911800: 1, 922577: 1, 925622: 1, 91979701: 1, 84610002: 1, 898431: 1,
                    893988: 1, 88466802: 1, 915452: 1, 8810158: 1, 8712766: 1, 908469: 1, 854002: 1, 919537: 1,
                    852973: 1, 861853: 1, 881046502: 1, 905189: 1, 91550: 1, 901088: 1, 911296202: 1, 8510426: 1,
                    894326: 1, 88147202: 1, 857010: 1, 868223: 1, 894855: 1, 869254: 1, 874373: 1, 9112366: 1,
                    8910721: 1, 8712064: 1, 926125: 1, 901041: 1, 87930: 1, 889719: 1, 913535: 1, 914862: 1, 865137: 1,
                    9113455: 1, 892438: 1, 91813701: 1, 873357: 1, 921644: 1, 884626: 1, 899987: 1, 866203: 1,
                    8910748: 1, 854941: 1, 85759902: 1, 91903902: 1, 908194: 1, 879523: 1, 901028: 1, 9113514: 1,
                    877486: 1, 861103: 1, 915186: 1, 892657: 1, 869104: 1, 911408: 1, 893526: 1, 875093: 1, 899667: 1,
                    922296: 1, 89511502: 1, 89511501: 1, 8813129: 1, 911296201: 1, 855625: 1, 852552: 1, 844359: 1,
                    883270: 1, 859717: 1, 897604: 1, 917080: 1, 875099: 1, 906878: 1, 914333: 1, 90745: 1, 886776: 1,
                    898677: 1, 908916: 1, 8910720: 1, 9110127: 1, 864877: 1, 925277: 1, 853612: 1, 915664: 1, 915143: 1,
                    861799: 1, 8610404: 1, 897630: 1, 859711: 1, 842302: 1, 889403: 1, 903507: 1, 848406: 1, 9112085: 1,
                    855563: 1, 901549: 1, 84501001: 1, 868871: 1, 919812: 1, 89263202: 1, 921092: 1, 862722: 1,
                    925291: 1, 9113156: 1, 85922302: 1, 862717: 1, 88147101: 1, 8712729: 1, 84799002: 1, 919555: 1,
                    9013005: 1, 86561: 1, 857637: 1, 868202: 1, 869931: 1, 911916: 1, 846381: 1, 8612399: 1,
                    88330202: 1, 925236: 1, 851509: 1, 88411702: 1, 917062: 1, 8711803: 1, 88725602: 1, 871149: 1,
                    905502: 1, 86135501: 1, 901836: 1, 90250: 1, 89382602: 1, 89382601: 1, 914580: 1, 88350402: 1,
                    8711002: 1, 857793: 1, 9010877: 1, 892604: 1, 922297: 1, 898143: 1, 926954: 1, 86135502: 1, 8913: 1,
                    92751: 1, 892189: 1, 905686: 1, 874217: 1, 9010872: 1, 8611555: 1, 924084: 1, 884448: 1, 877159: 1,
                    857810: 1, 917896: 1, 84458202: 1, 926424: 1, 884437: 1, 89812: 1, 85715: 1, 914102: 1, 885429: 1,
                    886452: 1, 905680: 1, 895633: 1, 8712853: 1, 864496: 1, 88143502: 1, 91789: 1, 894604: 1, 907914: 1,
                    89346: 1, 8912521: 1, 868999: 1, 921362: 1, 894335: 1, 903811: 1, 8711561: 1, 925311: 1, 907915: 1,
                    869224: 1, 852631: 1, 894618: 1, 909410: 1, 8511133: 1, 916838: 1, 8910499: 1, 891936: 1, 913063: 1,
                    89827: 1, 8910506: 1, 874158: 1, 914062: 1, 918192: 1, 872113: 1, 875878: 1
                }
            },
            'Diagnosis': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'B': 357, 'M': 212
                }
            },
            'Mean Radius': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.981, 'max': 28.11,
                    'mean': 14.127291739894563, 'median': 13.37, 'std': 3.524048826212078,
                    'skewness': 0.9423795716730992, 'kurtosis': 0.8455216229065377
                }
            },
            'Mean Texture': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 9.71, 'max': 39.28,
                    'mean': 19.28964850615117, 'median': 18.84, 'std': 4.301035768166949,
                    'skewness': 0.6504495420828159, 'kurtosis': 0.7583189723727752
                }
            },
            'Mean Perimeter': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 43.79, 'max': 188.5,
                    'mean': 91.96903339191566, 'median': 86.24, 'std': 24.2989810387549,
                    'skewness': 0.9906504253930081, 'kurtosis': 0.9722135477110654
                }
            },
            'Mean Area': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 143.5, 'max': 2501.0,
                    'mean': 654.8891036906857, 'median': 551.1, 'std': 351.9141291816527,
                    'skewness': 1.6457321756240424, 'kurtosis': 3.6523027623507582
                }
            },
            'Mean Smoothness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.052629999999999996, 'max': 0.1634,
                    'mean': 0.096360281195079, 'median': 0.09587000000000001, 'std': 0.014064128137673618,
                    'skewness': 0.45632376481956155, 'kurtosis': 0.8559749303632262
                }
            },
            'Mean Compactness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.01938, 'max': 0.3454,
                    'mean': 0.10434098418277686, 'median': 0.09262999999999999, 'std': 0.0528127579325122,
                    'skewness': 1.1901230311980404, 'kurtosis': 1.650130467219256
                }
            },
            'Mean Concavity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.4268,
                    'mean': 0.08879931581722322, 'median': 0.06154, 'std': 0.0797198087078935,
                    'skewness': 1.4011797389486722, 'kurtosis': 1.9986375291042124
                }
            },
            'Mean Concave Points': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.2012,
                    'mean': 0.048919145869947236, 'median': 0.0335, 'std': 0.03880284485915359,
                    'skewness': 1.1711800812336282, 'kurtosis': 1.066555702965477
                }
            },
            'Mean Symmetry': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.106, 'max': 0.304,
                    'mean': 0.181161862917399, 'median': 0.1792, 'std': 0.027414281336035712,
                    'skewness': 0.7256089733642002, 'kurtosis': 1.2879329922294565
                }
            },
            'Mean Fractal Dimension': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.049960000000000004, 'max': 0.09744,
                    'mean': 0.06279760984182776, 'median': 0.06154, 'std': 0.007060362795084458,
                    'skewness': 1.3044888125755076, 'kurtosis': 3.005892120169494
                }
            },
            'Radius SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1115, 'max': 2.873,
                    'mean': 0.4051720562390161, 'median': 0.3242, 'std': 0.2773127329861041,
                    'skewness': 3.088612166384756, 'kurtosis': 17.686725966164637
                }
            },
            'Texture SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.3602, 'max': 4.885,
                    'mean': 1.2168534270650269, 'median': 1.1079999999999999, 'std': 0.5516483926172023,
                    'skewness': 1.646443808753053, 'kurtosis': 5.349168692469973
                }
            },
            'Perimeter SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.757, 'max': 21.98,
                    'mean': 2.8660592267135288, 'median': 2.287, 'std': 2.021854554042107,
                    'skewness': 3.4436152021948976, 'kurtosis': 21.40190492588044
                }
            },
            'Area SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.8020000000000005, 'max': 542.2,
                    'mean': 40.33707908611603, 'median': 24.53, 'std': 45.49100551613178,
                    'skewness': 5.447186284898394, 'kurtosis': 49.20907650724119
                }
            },
            'Smoothness SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.001713, 'max': 0.03113,
                    'mean': 0.007040978910369071, 'median': 0.006379999999999999, 'std': 0.003002517943839067,
                    'skewness': 2.314450056636761, 'kurtosis': 10.469839532360393
                }
            },
            'Compactness SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.002252, 'max': 0.1354,
                    'mean': 0.025478138840070306, 'median': 0.02045, 'std': 0.017908179325677377,
                    'skewness': 1.9022207096378565, 'kurtosis': 5.10625248342338
                }
            },
            'Concavity SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.396,
                    'mean': 0.03189371634446394, 'median': 0.025889999999999996, 'std': 0.030186060322988394,
                    'skewness': 5.110463049043661, 'kurtosis': 48.8613953017919
                }
            },
            'Concave Points SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.05279,
                    'mean': 0.011796137082601056, 'median': 0.01093, 'std': 0.006170285174046866,
                    'skewness': 1.4446781446974788, 'kurtosis': 5.1263019430439565
                }
            },
            'Symmetry SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.007882, 'max': 0.07895,
                    'mean': 0.020542298769771532, 'median': 0.01873, 'std': 0.008266371528798399,
                    'skewness': 2.195132899547822, 'kurtosis': 7.896129827528971
                }
            },
            'Fractal Dimension SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0008948000000000001, 'max': 0.02984,
                    'mean': 0.0037949038664323374, 'median': 0.003187, 'std': 0.0026460709670891942,
                    'skewness': 3.923968620227413, 'kurtosis': 26.280847486373336
                }
            },
            'Worst Radius': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 7.93, 'max': 36.04,
                    'mean': 16.269189806678394, 'median': 14.97, 'std': 4.833241580469324,
                    'skewness': 1.1031152059604372, 'kurtosis': 0.9440895758772196
                }
            },
            'Worst Texture': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 12.02, 'max': 49.54,
                    'mean': 25.677223198594014, 'median': 25.41, 'std': 6.146257623038323,
                    'skewness': 0.49832130948716474, 'kurtosis': 0.22430186846478772
                }
            },
            'Worst Perimeter': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 50.41, 'max': 251.2,
                    'mean': 107.2612126537786, 'median': 97.66, 'std': 33.60254226903635,
                    'skewness': 1.1281638713683722, 'kurtosis': 1.070149666654432
                }
            },
            'Worst Area': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 185.2, 'max': 4254.0,
                    'mean': 880.5831282952545, 'median': 686.5, 'std': 569.3569926699492,
                    'skewness': 1.8593732724433467, 'kurtosis': 4.396394828992138
                }
            },
            'Worst Smoothness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.07117000000000001, 'max': 0.2226,
                    'mean': 0.13236859402460469, 'median': 0.1313, 'std': 0.022832429404835458,
                    'skewness': 0.4154259962824678, 'kurtosis': 0.5178251903311124
                }
            },
            'Worst Compactness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.02729, 'max': 1.058,
                    'mean': 0.25426504393673144, 'median': 0.2119, 'std': 0.15733648891374194,
                    'skewness': 1.4735549003297963, 'kurtosis': 3.0392881719200675
                }
            },
            'Worst Concavity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.252,
                    'mean': 0.27218848330404205, 'median': 0.2267, 'std': 0.20862428060813235,
                    'skewness': 1.1502368219460262, 'kurtosis': 1.6152532975830205
                }
            },
            'Worst Concave Points': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.29100000000000004,
                    'mean': 0.11460622319859404, 'median': 0.09992999999999999, 'std': 0.0657323411959421,
                    'skewness': 0.49261552688550875, 'kurtosis': -0.5355351225188589
                }
            },
            'Worst Symmetry': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1565, 'max': 0.6638,
                    'mean': 0.29007557117750454, 'median': 0.2822, 'std': 0.06186746753751869,
                    'skewness': 1.4339277651893279, 'kurtosis': 4.4445595178465815
                }
            },
            'Worst Fractal Dimension': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.05504, 'max': 0.2075,
                    'mean': 0.08394581722319855, 'median': 0.08004, 'std': 0.01806126734889399,
                    'skewness': 1.6625792663955172, 'kurtosis': 5.2446105558150125
                }
            }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 50 %
            #     'Diagnosis': {
            #         'B': 379, 'M': 190
            #     },
            #     'Mean Radius': {
            #         'min': 10.055320797652005, 'max': 19.731359430655836,
            #         'mean': 14.457030045774529, 'median': 13.654933206573128, 'std': 2.586464174879652,
            #         'skewness': 0.4375080151292858, 'kurtosis': -1.1719999429317418
            #     },
            #     'Mean Texture': {
            #         'min': 15.537648369073871, 'max': 23.035420253574852,
            #         'mean': 19.34465166796786, 'median': 19.05188373874873, 'std': 1.8030595013216215,
            #         'skewness': 0.14990584599600765, 'kurtosis': -1.0762088590148318
            #     },
            #     'Mean Perimeter': {
            #         'min': 62.77556610703468, 'max': 132.83638105541468,
            #         'mean': 92.96180594583635, 'median': 87.42389553695918, 'std': 17.973788502070324,
            #         'skewness': 0.4606778992637264, 'kurtosis': -1.1476514273266814
            #     },
            #     'Mean Area': {
            #         'min': 251.46817603707314, 'max': 1264.4593060538173,
            #         'mean': 680.6467858334127, 'median': 596.9211028814316, 'std': 261.5147544072258,
            #         'skewness': 0.5528283500562635, 'kurtosis': -1.0169019991555919
            #     },
            #     'Mean Smoothness': {
            #         'min': 0.08408950125440956, 'max': 0.11118742950960994,
            #         'mean': 0.09677928250017535, 'median': 0.09547357580330224, 'std': 0.006309907732357425,
            #         'skewness': 0.3042444038925176, 'kurtosis': -0.9955312626617117
            #     },
            #     'Mean Compactness': {
            #         'min': 0.04288180713236331, 'max': 0.1874190197342634,
            #         'mean': 0.10834946986406485, 'median': 0.09654436030715703, 'std': 0.03634624299603787,
            #         'skewness': 0.48676515666135994, 'kurtosis': -1.072819634533109
            #     },
            #     'Mean Concavity': {
            #         'min': 0.011091480243206026, 'max': 0.23034548698663715,
            #         'mean': 0.09299970759502547, 'median': 0.07135308474898339, 'std': 0.05758679593275987,
            #         'skewness': 0.6767312218118485, 'kurtosis': -0.8963628796369134
            #     },
            #     'Mean Concave Points': {
            #         'min': 0.009604415047168732, 'max': 0.11930115051865578,
            #         'mean': 0.05189332201930696, 'median': 0.04042334629893303, 'std': 0.030253097388888258,
            #         'skewness': 0.6119536871268991, 'kurtosis': -0.9945979390571709
            #     },
            #     'Mean Symmetry': {
            #         'min': 0.15839773269891735, 'max': 0.20699949173312632,
            #         'mean': 0.1806952448092904, 'median': 0.17807985063456, 'std': 0.010876172762667922,
            #         'skewness': 0.33465647873611254, 'kurtosis': -0.8981222474815866
            #     },
            #     'Mean Fractal Dimension': {
            #         'min': 0.05888641685217619, 'max': 0.06706644626185299,
            #         'mean': 0.06288961062540746, 'median': 0.06292028295993805, 'std': 0.0012728351538155637,
            #         'skewness': -0.02877270457697754, 'kurtosis': 0.0017005686183639845
            #     },
            #     'Radius SE': {
            #         'min': 0.15247100448608408, 'max': 0.9114652202576399,
            #         'mean': 0.41577836398352475, 'median': 0.3406434214115144, 'std': 0.1989038699717034,
            #         'skewness': 0.7776609658999765, 'kurtosis': -0.6430917406015952
            #     },
            #     'Texture SE': {
            #         'min': 0.9174099308013914, 'max': 1.6551471997261045,
            #         'mean': 1.225825906951417, 'median': 1.2194533658981321, 'std': 0.10347825347296515,
            #         'skewness': 0.30421905611347627, 'kurtosis': 0.41210279656950455
            #     },
            #     'Perimeter SE': {
            #         'min': 1.1037773499548444, 'max': 6.681291440591218,
            #         'mean': 2.8927387252243304, 'median': 2.4269869525909433, 'std': 1.3463109100593635,
            #         'skewness': 0.7653611037258816, 'kurtosis': -0.5693434485057605
            #     },
            #     'Area SE': {
            #         'min': 9.996555586040023, 'max': 141.30952972078325,
            #         'mean': 42.67611590189107, 'median': 30.866472198486314, 'std': 29.305224362991257,
            #         'skewness': 1.0484678059886061, 'kurtosis': 0.12954167403138017
            #     },
            #     'Smoothness SE': {
            #         'min': 0.005487707497596741, 'max': 0.009921876399755478,
            #         'mean': 0.00726363189483705, 'median': 0.007201313035055994, 'std': 0.0006716369962939125,
            #         'skewness': 0.8522551798134925, 'kurtosis': 1.3620578574220215},
            #     'Compactness SE': {
            #         'min': 0.01217927109515667, 'max': 0.048041578226625914,
            #         'mean': 0.026753470448370696, 'median': 0.024841890962958333, 'std': 0.00829457756520085,
            #         'skewness': 0.46696289708446675, 'kurtosis': -0.8608379972234608},
            #     'Concavity SE': {
            #         'min': 0.003847041964530945, 'max': 0.08402678060531617,
            #         'mean': 0.031876416612656146, 'median': 0.026505172061920167, 'std': 0.01773728600010749,
            #         'skewness': 0.72281983444258, 'kurtosis': -0.48127110299951514
            #     },
            #     'Concave Points SE': {
            #         'min': 0.006086186686754227, 'max': 0.01889403624355793,
            #         'mean': 0.012097012599344597, 'median': 0.011398884126543999, 'std': 0.0030969736550066364,
            #         'skewness': 0.3377073489500882, 'kurtosis': -0.9523443971857328
            #     },
            #     'Symmetry SE': {
            #         'min': 0.015651154805779454, 'max': 0.02807586978918314,
            #         'mean': 0.02109623830601643, 'median': 0.02116888470292091, 'std': 0.0017570133969569397,
            #         'skewness': -0.14793523275675455, 'kurtosis': 0.4989044508824483
            #     },
            #     'Fractal Dimension SE': {
            #         'min': 0.0015554578772425636, 'max': 0.00837074408892393,
            #         'mean': 0.0038666474313793137, 'median': 0.0035903526017367834, 'std': 0.0013467201022157986,
            #         'skewness': 0.8222042577260034, 'kurtosis': 0.27231300997602137
            #     },
            #     'Worst Radius': {
            #         'min': 10.832956399321557, 'max': 24.6085958224535,
            #         'mean': 16.42518882830817, 'median': 15.297093951255084, 'std': 3.625844086472233,
            #         'skewness': 0.5292152345009977, 'kurtosis': -1.0759560114364461
            #     },
            #     'Worst Texture': {
            #         'min': 20.2737514090538, 'max': 31.51827042177319,
            #         'mean': 25.83229640004886, 'median': 25.21270367056131, 'std': 2.832023519010256,
            #         'skewness': 0.19177969380493562, 'kurtosis': -1.1499238678363877
            #     },
            #     'Worst Perimeter': {
            #         'min': 69.8011197528243, 'max': 165.35209877498446,
            #         'mean': 108.71814272687034, 'median': 100.56831517487764, 'std': 26.30021453629827,
            #         'skewness': 0.45474851817881756, 'kurtosis': -1.2091634978616712
            #     },
            #     'Worst Area': {
            #         'min': 308.0208345293999, 'max': 1913.0535159133376,
            #         'mean': 921.0553290893413, 'median': 741.325099182129, 'std': 441.4549382707825,
            #         'skewness': 0.6500916453396487, 'kurtosis': -0.9728807433461859
            #     },
            #     'Worst Smoothness': {
            #         'min': 0.11203179253503681, 'max': 0.15442371793013066,
            #         'mean': 0.13350505922392342, 'median': 0.13131642450653017, 'std': 0.010376605478351038,
            #         'skewness': 0.24351100897149783, 'kurtosis': -1.1526861660374803
            #     },
            #     'Worst Compactness': {
            #         'min': 0.09996634862363334, 'max': 0.48707066073171795,
            #         'mean': 0.2592581410208061, 'median': 0.22831706340074537, 'std': 0.10366176096790636,
            #         'skewness': 0.5424595539505415, 'kurtosis': -1.0120652182959118
            #     },
            #     'Worst Concavity': {
            #         'min': 0.06016527414321899, 'max': 0.6286894740164279,
            #         'mean': 0.2842507352572524, 'median': 0.2357809818983078, 'std': 0.1482035395363268,
            #         'skewness': 0.5503619046777104, 'kurtosis': -1.073338238316828
            #     },
            #     'Worst Concave Points': {
            #         'min': 0.029398724207282064, 'max': 0.22113341622054575,
            #         'mean': 0.11431064853593904, 'median': 0.09752719004005193, 'std': 0.05262539330144273,
            #         'skewness': 0.3921411844663567, 'kurtosis': -1.290847234360129
            #     },
            #     'Worst Symmetry': {
            #         'min': 0.23579858380556104, 'max': 0.3562747389174998,
            #         'mean': 0.2932149158116374, 'median': 0.28885915210247043, 'std': 0.026574616860505725,
            #         'skewness': 0.22735971678937597, 'kurtosis': -0.9306362653147371
            #     },
            #     'Worst Fractal Dimension': {
            #         'min': 0.06901861355662345, 'max': 0.10405269536003471,
            #         'mean': 0.08466764643454303, 'median': 0.08266884924471378, 'std': 0.007946024163687814,
            #         'skewness': 0.3729168838206147, 'kurtosis': -0.9121885849961928
            #     }
            #  }
            ############################################################################################################
        },      # ok
        'credit': {

        },
        'eeg': {  # https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
            'AF3': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1030.77, 'max': 309231.0, 'mean': 4321.917777036056, 'median': 4294.36,
                    'std': 2492.0721742651103, 'skewness': 122.29386525011812, 'kurtosis': 14963.840002182125
                }
            },
            'F7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2830.77, 'max': 7804.62, 'mean': 4009.767693591461, 'median': 4005.64,
                    'std': 45.94167248479191, 'skewness': 39.046557690711396, 'kurtosis': 3210.171915006226
                }
            },
            'F3': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1040.0, 'max': 6880.51, 'mean': 4264.022432576795, 'median': 4262.56,
                    'std': 44.428051757419446, 'skewness': -13.615160740497625, 'kurtosis': 2921.967694389361
                }
            },
            'FC5': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2453.33, 'max': 642564.0, 'mean': 4164.9463264352735, 'median': 4120.51,
                    'std': 5216.40463229992, 'skewness': 122.38777688436551, 'kurtosis': 14979.17873537246
                }
            },
            'T7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2089.74, 'max': 6474.36, 'mean': 4341.741075433922, 'median': 4338.97,
                    'std': 34.73882081848658, 'skewness': 7.561902122619299, 'kurtosis': 2578.229693199016
                }
            },
            'P7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2768.21, 'max': 362564.0, 'mean': 4644.022379172214, 'median': 4617.95,
                    'std': 2924.7895373250954, 'skewness': 122.36281054964749, 'kurtosis': 14975.088891085208
                }
            },
            'O1': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2086.15, 'max': 567179.0, 'mean': 4110.400159546061, 'median': 4070.26,
                    'std': 4600.926542533738, 'skewness': 122.38359282836112, 'kurtosis': 14978.495281856303
                }
            },
            'O2': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4567.18, 'max': 7264.1, 'mean': 4616.056903871852, 'median': 4613.33,
                    'std': 29.292603201776014, 'skewness': 51.09721901768454, 'kurtosis': 4491.11404630705
                }
            },
            'P8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1357.95, 'max': 265641.0, 'mean': 4218.826610146869, 'median': 4199.49,
                    'std': 2136.4085228873855, 'skewness': 122.33467120005805, 'kurtosis': 14970.509845636481
                }
            },
            'T8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1816.41, 'max': 6674.36, 'mean': 4231.316199599468, 'median': 4229.23,
                    'std': 38.05090262121652, 'skewness': 10.23070102200852, 'kurtosis': 2710.083429497155
                }
            },
            'FC6': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 3273.33, 'max': 6823.08, 'mean': 4202.456899866489, 'median': 4200.51,
                    'std': 37.78598137403701, 'skewness': 31.649004824348655, 'kurtosis': 2056.5210594418677
                }
            },
            'F4': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2257.95, 'max': 7002.56, 'mean': 4279.232774365836, 'median': 4276.92,
                    'std': 41.54431151666411, 'skewness': 26.556468850889043, 'kurtosis': 2714.7186392430435
                }
            },
            'F8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 86.6667, 'max': 152308.0, 'mean': 4615.205335560761, 'median': 4603.08,
                    'std': 1208.3699582560462, 'skewness': 121.90727242585102, 'kurtosis': 14901.910996495131
                }
            },
            'AF4': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1366.15, 'max': 715897.0, 'mean': 4416.435832443256, 'median': 4354.87,
                    'std': 5891.2850425236575, 'skewness': 118.1250449998719, 'kurtosis': 14214.276393221251
                }
            },
            'Eye Detection': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {0: 8257, 1: 6723}
            }
        },         # ok
        'iris': {  # https://archive.ics.uci.edu/ml/datasets/Iris
            'sepal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4.3, 'max': 7.9, 'mean': 5.843333333333335, 'median': 5.8,
                    'std': 0.8280661279778629, 'skewness': 0.3149109566369728, 'kurtosis': -0.5520640413156395
                }
            },
            'sepal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.0, 'max': 4.4, 'mean': 3.0540000000000007, 'median': 3.0,
                    'std': 0.4335943113621737, 'skewness': 0.3340526621720866, 'kurtosis': 0.2907810623654279
                }
            },
            'petal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 6.9, 'mean': 3.7586666666666693, 'median': 4.35,
                    'std': 1.7644204199522617, 'skewness': -0.27446425247378287, 'kurtosis': -1.4019208006454036
                }
            },
            'petal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1, 'max': 2.5, 'mean': 1.1986666666666672, 'median': 1.3,
                    'std': 0.7631607417008414, 'skewness': -0.10499656214412734, 'kurtosis': -1.3397541711393433
                }
            },
            'class': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Iris-virginica': 50, 'Iris-versicolor': 50, 'Iris-setosa': 50
                }
            }
            # {  # SynSGAIN --> missing rate = 10%
            #     'sepal length': {
            #         'min': 4.869143724441528, 'max': 6.756005132198334,
            #         'mean': 5.865372681950918, 'median': 5.987367056310177,
            #         'std': 0.6443288269812552, 'skewness': -0.2672105514743698, 'kurtosis': -1.440892477693148
            #     },
            #     'sepal width': {
            #         'min': 2.6794557571411133, 'max': 3.4605865299701692,
            #         'mean': 3.049511124604693, 'median': 3.018159952759743,
            #         'std': 0.24667888191140247, 'skewness': 0.24891297495248887, 'kurtosis': -1.2941566744870767
            #     },
            #     'petal length': {
            #         'min': 1.3751519292593002, 'max': 5.850026854872705,
            #         'mean': 3.7500563668540168, 'median': 4.294739981926978,
            #         'std': 1.5743475349621674, 'skewness': -0.41351663348677464, 'kurtosis': -1.4298252871716108
            #     },
            #     'petal width': {
            #         'min': 0.24593613147735585, 'max': 2.2246293306350706,
            #         'mean': 1.2159415396172553, 'median': 1.340029091760516,
            #         'std': 0.6624534469169271, 'skewness': -0.2667101173670794, 'kurtosis': -1.4024854893610996
            #     },
            #     'class': {'Iris-setosa': 55, 'Iris-virginica': 49, 'Iris-versicolor': 46}
            # }
            #
            # {  # SynSGAIN --> missing rate = 50%
            #     'sepal length': {
            #         'min': 4.725000470876694, 'max': 7.159991952776909,
            #         'mean': 5.91142641463845, 'median': 6.027985951304435,
            #         'std': 0.6162567263864119, 'skewness': -0.3685339349358077, 'kurtosis': -0.7062707573537965
            #     },
            #     'sepal width': {
            #         'min': 2.5121456384658813, 'max': 4.0,
            #         'mean': 3.1123621002460524, 'median': 3.0,
            #         'std': 0.2727797843659712, 'skewness': 1.3105421746575598, 'kurtosis': 2.5107829060285045
            #     },
            #     'petal length': {
            #         'min': 1.1178509563207626, 'max': 6.324104183912278,
            #         'mean': 3.734171063112095, 'median': 4.2021530896425245,
            #         'std': 1.4240954337825102, 'skewness': -0.45109119516595103, 'kurtosis': -0.9806346565718504
            #     },
            #     'petal width': {
            #         'min': 0.12071516513824454, 'max': 2.3104502916336056,
            #         'mean': 1.1839774532690641, 'median': 1.3267868269234895,
            #         'std': 0.6361145328961184, 'skewness': -0.29860962394272095, 'kurtosis': -1.0270348998276746
            #     },
            #     'class': {'Iris-setosa': 73, 'Iris-versicolor': 41, 'Iris-virginica': 36}
            #  }
        },        # ok
        'letter': {

        },
        'mushroom': {  # https://archive.ics.uci.edu/ml/datasets/Mushroom
            'class': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e', 4208, 'p', 3916
                }
            },
            'cap-shape': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 452, 'c', 4, 'f', 3152, 'k', 828, 's', 32, 'x', 3656
                }
            },
            'cap-surface': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 2320, 'g', 4, 's', 2556, 'y', 3244
                }
            },
            'cap-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 168, 'c', 44, 'e', 1500, 'g', 1840, 'n', 2284, 'p', 144, 'r', 16, 'u', 16, 'w', 1040, 'y', 1072
                }
            },
            'bruises': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f': 4748, 't': 3376
                }
            },
            'odor': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a', 400, 'c', 192, 'f', 2160, 'l', 400, 'm', 36, 'n', 3528, 'p', 256, 's', 576, 'y', 576
                }
            },
            'gill-attachment': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a': 210, 'f': 7914
                }
            },
            'gill-spacing': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'c': 6812, 'w': 1312
                }
            },
            'gill-size': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b': 5612, 'n': 2512
                }
            },
            'gill-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 1728, 'e', 96, 'g', 752, 'h', 732, 'k', 408, 'n', 1048,
                    'o', 64, 'p', 1492, 'r', 24, 'u', 492, 'w', 1202, 'y', 86
                }
            },
            'stalk-shape': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e': 3516, 't': 4608
                }
            },
            'stalk-root': {  # drop feature/variable --> has missing values
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': ['?'],
                'values_dist': {
                    'b': 3776, 'c': 556, 'e': 1120, 'r': 192, '?': 2480
                }
            },
            'stalk-surface-above-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 552, 'k', 2372, 's', 5176, 'y', 24
                }
            },
            'stalk-surface-below-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 600, 'k', 2304, 's', 4936, 'y', 284
                }
            },
            'stalk-color-above-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 432, 'c', 36, 'e', 96, 'g', 576, 'n', 448, 'o', 192, 'p', 1872, 'w', 4464, 'y', 8
                }
            },
            'stalk-color-below-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 432, 'c', 36, 'e', 96, 'g', 576, 'n', 512, 'o', 192, 'p', 1872, 'w', 4384, 'y', 24
                }
            },
            'veil-type': {  # this feature contributes with NOTHING to any ML task --> variance = 0
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    'p': 8124
                }
            },
            'veil-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'n', 96, 'o', 96, 'w', 7924, 'y', 8
                }
            },
            'ring-number': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'n', 36, 'o', 7488, 't', 600
                }
            },
            'ring-type': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e', 2776, 'f', 48, 'l', 1296, 'n', 36, 'p', 3968
                }
            },
            'spore-print-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 48, 'h', 1632, 'k', 1872, 'n', 1968, 'o', 48, 'r', 72, 'u', 48, 'w', 2388, 'y', 48
                }
            },
            'population': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a', 384, 'c', 340, 'n', 400, 's', 1248, 'v', 4040, 'y', 1712
                }
            },
            'habitat': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'd', 3148, 'g', 2148, 'l', 832, 'm', 292, 'p', 1144, 'u', 368, 'w', 192
                }
            }
        },    # ok
        'news': {

        },
        'spam': {  # https://archive.ics.uci.edu/ml/datasets/Spambase

        },
        'wine-red': {  # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
            'fixed acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4.6, 'max': 15.9,
                    'mean': 8.319637273295838, 'median': 7.9, 'std': 1.7410963181277006,
                    'skewness': 0.9827514413284587, 'kurtosis': 1.1321433977276252
                }
            },
            'volatile acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.12, 'max': 1.58,
                    'mean': 0.5278205128205131, 'median': 0.52, 'std': 0.17905970415353498,
                    'skewness': 0.6715925723840199, 'kurtosis': 1.2255422501791422
                }
            },
            'citric acid': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.2709756097560964, 'median': 0.26, 'std': 0.19480113740531785,
                    'skewness': 0.3183372952546368, 'kurtosis': -0.7889975153633966
                }
            },
            'residual sugar': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.9, 'max': 15.5,
                    'mean': 2.5388055034396517, 'median': 2.2, 'std': 1.4099280595072805,
                    'skewness': 4.54065542590319, 'kurtosis': 28.617595424475443
                }
            },
            'chlorides': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.012, 'max': 0.611,
                    'mean': 0.08746654158849257, 'median': 0.079, 'std': 0.047065302010090154,
                    'skewness': 5.680346571971722, 'kurtosis': 41.71578724757661
                }
            },
            'free sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 72.0,
                    'mean': 15.874921826141339, 'median': 14.0, 'std': 10.46015696980973,
                    'skewness': 1.250567293314441, 'kurtosis': 2.023562045840575
                }
            },
            'total sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.0, 'max': 289.0,
                    'mean': 46.46779237023139, 'median': 38.0, 'std': 32.89532447829901,
                    'skewness': 1.515531257594554, 'kurtosis': 3.8098244878645744
                }
            },
            'density': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.9900700000000001, 'max': 1.00369,
                    'mean': 0.9967466791744833, 'median': 0.99675, 'std': 0.0018873339538425563,
                    'skewness': 0.07128766294945525, 'kurtosis': 0.9340790654648083
                }
            },
            'pH': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.74, 'max': 4.01,
                    'mean': 3.311113195747343, 'median': 3.31, 'std': 0.15438646490354266,
                    'skewness': 0.19368349811284427, 'kurtosis': 0.806942508246574
                }
            },
            'sulphates': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.33, 'max': 2.0,
                    'mean': 0.6581488430268921, 'median': 0.62, 'std': 0.16950697959010977,
                    'skewness': 2.4286723536602945, 'kurtosis': 11.720250727147674
                }
            },
            'alcohol': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 8.4, 'max': 14.9,
                    'mean': 10.422983114446502, 'median': 10.2, 'std': 1.0656675818473926,
                    'skewness': 0.8608288068888538, 'kurtosis': 0.2000293113417695
                }
            },
            'quality': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    5: 681, 6: 638, 7: 199, 4: 53, 8: 18, 3: 10
                }
            }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 10 %
            #     'fixed acidity': {
            #         'min': 7.638615542650222, 'max': 10.220858364645391,
            #         'mean': 8.41480888617669, 'median': 8.322367368638515, 'std': 0.3957990055342743,
            #         'skewness': 1.4102288904501026, 'kurtosis': 2.087524292746375
            #     },
            #     'volatile acidity': {
            #         'min': 0.4079621690511704, 'max': 0.699630975574255,
            #         'mean': 0.5433102018184369, 'median': 0.5365832975506782, 'std': 0.053122966438067916,
            #         'skewness': -0.12200095495276808, 'kurtosis': -0.9118514187995896
            #     },
            #     'citric acid': {
            #         'min': 0.21173596382141113, 'max': 0.4750596385449172,
            #         'mean': 0.2830710635071717, 'median': 0.2681887745857239, 'std': 0.05505055281777822,
            #         'skewness': 1.4713539368860615, 'kurtosis': 1.535044929952484
            #     },
            #     'residual sugar': {
            #         'min': 2.0982131958007817, 'max': 4.240120628476143,
            #         'mean': 2.6950196501965054, 'median': 2.6197572588920597, 'std': 0.32758729738372433,
            #         'skewness': 1.2328382469717976, 'kurtosis': 1.601493406807358
            #     },
            #     'chlorides': {
            #         'min': 0.06578971812129022, 'max': 0.14947163993120197,
            #         'mean': 0.09198044101630166, 'median': 0.09050494253635408, 'std': 0.011960203908899458,
            #         'skewness': 0.7721836867287272, 'kurtosis': 0.7980028673815345
            #     },
            #     'free sulfur dioxide': {
            #         'min': 13.472533881664274, 'max': 20.559005036950108,
            #         'mean': 16.59603975499847, 'median': 16.576406806707382, 'std': 1.2661714676122902,
            #         'skewness': 0.21701117606723308, 'kurtosis': -0.149412162201787
            #     },
            #     'total sulfur dioxide': {
            #         'min': 31.42834520339967, 'max': 82.56415230035783,
            #         'mean': 49.48263507831687, 'median': 50.550343394279494, 'std': 9.805251355017436,
            #         'skewness': 0.06399372886086356, 'kurtosis': -1.0208706073367775
            #     },
            #     'density': {
            #         'min': 0.9958441019083559, 'max': 0.9974680528961122,
            #         'mean': 0.9967702163893892, 'median': 0.9967439903737232, 'std': 0.0003179142672856969,
            #         'skewness': -0.4116937965201362, 'kurtosis': -0.3336054688736261
            #     },
            #     'pH': {
            #         'min': 3.217035687416792, 'max': 3.38779374091886,
            #         'mean': 3.3176488726725144, 'median': 3.319546595811844, 'std': 0.019707260578549112,
            #         'skewness': -0.6871705528954715, 'kurtosis': 2.120503793454242
            #     },
            #     'sulphates': {
            #         'min': 0.5716403743624686, 'max': 0.8487048329412936,
            #         'mean': 0.673065677406529, 'median': 0.6670353302359581, 'std': 0.051725990756231924,
            #         'skewness': 0.6441926878404182, 'kurtosis': 0.04127422928052393
            #     },
            #     'alcohol': {
            #         'min': 9.749462553858757, 'max': 11.788065204024315,
            #         'mean': 10.492501342456338, 'median': 10.55088214725256, 'std': 0.5125686993353608,
            #         'skewness': 0.48890525301147253, 'kurtosis': -0.5365889044331897
            #     },
            #     'quality': {
            #         5: 786, 6: 622, 3: 161, 7: 30
            #     }
            # }
            # {  # SynSGAIN --> missing rate = 50 %
            #     'fixed acidity': {
            #         'min': 7.574172461032867, 'max': 9.46866299882531,
            #         'mean': 8.46085231192946, 'median': 8.450857949256896, 'std': 0.2941094545003602,
            #         'skewness': 0.1967582776911927, 'kurtosis': -0.23967452409712253
            #     },
            #     'volatile acidity': {
            #         'min': 0.4174632441997528, 'max': 0.6523298723623157,
            #         'mean': 0.5375433252454079, 'median': 0.5398475641012191, 'std': 0.04308092234252141,
            #         'skewness': -0.07216691867079991, 'kurtosis': -0.36971320203268343
            #     },
            #     'citric acid': {
            #         'min': 0.18331998586654663, 'max': 0.3993986626295373,
            #         'mean': 0.28472602896713206, 'median': 0.2818143685162068, 'std': 0.0412236186094129,
            #         'skewness': 0.3364065231362639, 'kurtosis': -0.3205529666524818
            #     },
            #     'residual sugar': {
            #         'min': 1.8187778234481815, 'max': 4.54097660779953,
            #         'mean': 2.708326951271301, 'median': 2.6772880196571354, 'std': 0.3980952093859446,
            #         'skewness': 0.45421277124603976, 'kurtosis': 0.017674471796422786
            #     },
            #     'chlorides': {
            #         'min': 0.053873424023389835, 'max': 0.16094384574890136,
            #         'mean': 0.09385150421314334, 'median': 0.0926021556854248, 'std': 0.015493848928510438,
            #         'skewness': 0.4443555006225287, 'kurtosis': 0.10289235950280062
            #     },
            #     'free sulfur dioxide': {
            #         'min': 11.352267980575565, 'max': 20.98111987113953,
            #         'mean': 16.375636191695015, 'median': 16.356944441795353, 'std': 1.5720994304283713,
            #         'skewness': 0.0739562917451341, 'kurtosis': -0.28142520717681574
            #     },
            #     'total sulfur dioxide': {
            #         'min': 24.533292561769496, 'max': 80.04385779798032,
            #         'mean': 49.57704308736206, 'median': 49.62008666992188, 'std': 9.244960389985104,
            #         'skewness': 0.10541976324230723, 'kurtosis': -0.20843517876232154
            #     },
            #     'density': {
            #         'min': 0.9961357125274837, 'max': 0.9975675470465422,
            #         'mean': 0.9968980283834589, 'median': 0.9969031271168542, 'std': 0.0003384226889033401,
            #         'skewness': 0.04238092094749888, 'kurtosis': -0.8401345735894665
            #     },
            #     'pH': {
            #         'min': 3.258320507705212, 'max': 3.3607517145574093,
            #         'mean': 3.308323289075979, 'median': 3.3071180180087687, 'std': 0.01822952410433691,
            #         'skewness': 0.5372711842576674, 'kurtosis': 0.4377725861281969
            #     },
            #     'sulphates': {
            #         'min': 0.5456597077846527, 'max': 0.8510048630833624,
            #         'mean': 0.6789349350367678, 'median': 0.6783328700065612, 'std': 0.04929788323818455,
            #         'skewness': 0.15252611985586492, 'kurtosis': -0.28350828744593715
            #     },
            #     'alcohol': {
            #         'min': 9.502634072303772, 'max': 11.503164899349212,
            #         'mean': 10.498861524906996, 'median': 10.52743228673935, 'std': 0.4283711166146795,
            #         'skewness': -0.04852549510145854, 'kurtosis': -0.7615197982050721
            #     },
            #     'quality': {3: 751, 5: 463, 6: 346, 7: 39}
            #  }
            ############################################################################################################
        },    # ok
        'wine-white': {  # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
            'fixed acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 3.8, 'max': 14.2,
                    'mean': 6.854787668436075, 'median': 6.8, 'std': 0.8438682276875188,
                    'skewness': 0.6477514746297539, 'kurtosis': 2.1721784645585807
                }
            },
            'volatile acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.08, 'max': 1.1,
                    'mean': 0.27824111882401087, 'median': 0.26, 'std': 0.10079454842486428,
                    'skewness': 1.5769795029952025, 'kurtosis': 5.091625816866611
                }
            },
            'citric acid': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.66,
                    'mean': 0.33419150673743736, 'median': 0.32, 'std': 0.12101980420298301,
                    'skewness': 1.2819203981671066, 'kurtosis': 6.174900656983394
                }
            },
            'residual sugar': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.6, 'max': 65.8,
                    'mean': 6.391414863209486, 'median': 5.2, 'std': 5.072057784014864,
                    'skewness': 1.0770937564240868, 'kurtosis': 3.4698201025634265
                }
            },
            'chlorides': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.009000000000000001, 'max': 0.34600000000000003,
                    'mean': 0.0457723560636995, 'median': 0.043, 'std': 0.02184796809372882,
                    'skewness': 5.023330682759707, 'kurtosis': 37.564599706679516
                }
            },
            'free sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.0, 'max': 289.0,
                    'mean': 35.30808493262556, 'median': 34.0, 'std': 17.007137325232566,
                    'skewness': 1.4067449205303078, 'kurtosis': 11.466342426607905
                }
            },
            'total sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 9.0, 'max': 440.0,
                    'mean': 138.36065741118824, 'median': 134.0, 'std': 42.49806455414294,
                    'skewness': 0.3907098416536745, 'kurtosis': 0.5718532333534614
                }
            },
            'density': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.98711, 'max': 1.03898,
                    'mean': 0.9940273764801896, 'median': 0.99374, 'std': 0.0029909069169369393,
                    'skewness': 0.9777730048689881, 'kurtosis': 9.793806910765209
                }
            },
            'pH': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.72, 'max': 3.82,
                    'mean': 3.1882666394446693, 'median': 3.18, 'std': 0.1510005996150667,
                    'skewness': 0.4577825459180807, 'kurtosis': 0.5307749515326159
                }
            },
            'sulphates': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.22, 'max': 1.08,
                    'mean': 0.4898468762760325, 'median': 0.47, 'std': 0.11412583394883138,
                    'skewness': 0.9771936833065663, 'kurtosis': 1.5909296303516225
                }
            },
            'alcohol': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 8.0, 'max': 14.2,
                    'mean': 10.514267047774638, 'median': 10.4, 'std': 1.2306205677573183,
                    'skewness': 0.4873419932161276, 'kurtosis': -0.6984253277895518
                }
            },
            'quality': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    6: 2198, 5: 1457, 7: 880, 8: 175, 4: 163, 3: 20, 9: 5
                }
            }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 50 %
            #     'fixed acidity': {
            #         'min': 6.442024093866348, 'max': 7.487171474099159,
            #         'mean': 6.932601444685553, 'median': 6.929616558551788, 'std': 0.1349418141553136,
            #         'skewness': 0.14728806874637948, 'kurtosis': 0.601371120975303
            #     },
            #     'volatile acidity': {
            #         'min': 0.20721410334110255, 'max': 0.3762910619378089,
            #         'mean': 0.2848153598224937, 'median': 0.28488950490951537, 'std': 0.025275081284385208,
            #         'skewness': 0.06363665058291605, 'kurtosis': -0.14625629575526355
            #     },
            #     'citric acid': {
            #         'min': 0.2554733031988144, 'max': 0.47216584175825116,
            #         'mean': 0.34539537650321206, 'median': 0.34248533844947815, 'std': 0.02695777469285683,
            #         'skewness': 0.6394532304799541, 'kurtosis': 1.0973367381629187
            #     },
            #     'residual sugar': {
            #         'min': 3.178298997879025, 'max': 14.183473885059353,
            #         'mean': 6.852141280402285, 'median': 6.745217183977367, 'std': 1.376842382646811,
            #         'skewness': 0.7055840002925869, 'kurtosis': 1.1366853414821838
            #     },
            #     'chlorides': {
            #         'min': 0.02721017932891847, 'max': 0.0865727574825287,
            #         'mean': 0.04989603802562376, 'median': 0.04927048462629317, 'std': 0.008087627174594582,
            #         'skewness': 0.45013777674376154, 'kurtosis': 0.24383209121603544
            #     },
            #     'free sulfur dioxide': {
            #         'min': 20.60700523853303, 'max': 68.40471732616426,
            #         'mean': 37.89085987391594, 'median': 37.63522792235016, 'std': 5.549428653519843,
            #         'skewness': 0.5955439770564425, 'kurtosis': 1.9864687552016544
            #     },
            #     'total sulfur dioxide': {
            #         'min': 109.05916833877562, 'max': 166.03216382861135,
            #         'mean': 139.11109560796635, 'median': 139.2098967395723, 'std': 9.343837517018073,
            #         'skewness': -0.05356518428666135, 'kurtosis': -0.4317469678931918
            #     },
            #     'density': {
            #         'min': 0.990953530278206, 'max': 0.9992901488733292,
            #         'mean': 0.9944121083264795, 'median': 0.9943294990122319, 'std': 0.0011327671303058748,
            #         'skewness': 0.2976480704011007, 'kurtosis': 0.33021792470867384
            #     },
            #     'pH': {
            #         'min': 3.1315512216091155, 'max': 3.238586517125368,
            #         'mean': 3.181916226984109, 'median': 3.1822924974560736, 'std': 0.017308794641401632,
            #         'skewness': -0.04262876323679676, 'kurtosis': -0.41308690141612114
            #     },
            #     'sulphates': {
            #         'min': 0.44427878975868224, 'max': 0.5395834985375405,
            #         'mean': 0.49567815602063453, 'median': 0.49562682099640365, 'std': 0.013879735310700305,
            #         'skewness': -0.012669374249156467, 'kurtosis': -0.2649282488738365
            #     },
            #     'alcohol': {
            #         'min': 9.617617982625962, 'max': 11.801105190813542,
            #         'mean': 10.627598678830271, 'median': 10.65086571201682, 'std': 0.4711980936477008,
            #         'skewness': 0.17423929651304143, 'kurtosis': -0.4213432694382351
            #     },
            #     'quality': {
            #         3: 2191, 6: 1242, 5: 1154, 7: 311
            #     }
            # }
            ############################################################################################################
        },  # ok
        'yeast': {  # https://archive.ics.uci.edu/ml/datasets/Yeast
            'Sequence_Name': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'RL12_YEAST': 2, 'RS28_YEAST': 2, 'MTC_YEAST ': 2, 'RL2_YEAST ': 2, 'H3_YEAST  ': 2,
                    'RLUB_YEAST': 2, 'IF4A_YEAST': 2, 'RL44_YEAST': 2, 'RL41_YEAST': 2, 'MAT2_YEAST': 2,
                    'RL19_YEAST': 2, 'RS41_YEAST': 2, 'RS22_YEAST': 2, 'H4_YEAST  ': 2, 'RS24_YEAST': 2,
                    'RL15_YEAST': 2, 'RS8_YEAST ': 2, 'RL35_YEAST': 2, 'RL1A_YEAST': 2, 'EF1A_YEAST': 2,
                    'RS4E_YEAST': 2, 'RL18_YEAST': 2, 'DHSA_YEAST': 1, 'RL3A_YEAST': 1, 'RAS1_YEAST': 1,
                    'PES4_YEAST': 1, 'RS6_YEAST ': 1, 'R14A_YEAST': 1, 'BGL2_YEAST': 1, 'YCT8_YEAST': 1,
                    'SCO1_YEAST': 1, 'TAL1_YEAST': 1, 'MEIX_YEAST': 1, 'FU34_YEAST': 1, 'CC7_YEAST ': 1,
                    'MASZ_YEAST': 1, 'DPOX_YEAST': 1, 'R26B_YEAST': 1, 'EF1H_YEAST': 1, 'PUR4_YEAST': 1,
                    'YHK5_YEAST': 1, 'PROB_YEAST': 1, 'YB00_YEAST': 1, 'UBC2_YEAST': 1, 'E2BA_YEAST': 1,
                    'CYPR_YEAST': 1, 'NAB3_YEAST': 1, 'NOT1_YEAST': 1, 'SWI3_YEAST': 1, 'RA54_YEAST': 1,
                    'ADT2_YEAST': 1, 'SC11_YEAST': 1, 'PPAC_YEAST': 1, 'KR11_YEAST': 1, 'SC17_YEAST': 1,
                    'ADA2_YEAST': 1, 'HXKA_YEAST': 1, 'DPM1_YEAST': 1, 'YN94_YEAST': 1, 'PEM2_YEAST': 1,
                    'R13A_YEAST': 1, 'ACT2_YEAST': 1, 'YHU0_YEAST': 1, 'RTG1_YEAST': 1, 'PPX1_YEAST': 1,
                    'ACT3_YEAST': 1, 'R104_YEAST': 1, 'ERS1_YEAST': 1, 'CHL4_YEAST': 1, 'ARD1_YEAST': 1,
                    'SR54_YEAST': 1, 'RT05_YEAST': 1, 'THDH_YEAST': 1, 'M101_YEAST': 1, 'CCR4_YEAST': 1,
                    'TFC3_YEAST': 1, 'R161_YEAST': 1, 'RNC1_YEAST': 1, 'LAS1_YEAST': 1, 'SC23_YEAST': 1,
                    'MLH1_YEAST': 1, 'VATL_YEAST': 1, 'SNQ2_YEAST': 1, 'KEM1_YEAST': 1, 'YHQ1_YEAST': 1,
                    'N116_YEAST': 1, 'MRS3_YEAST': 1, 'SLA1_YEAST': 1, 'YGL1_YEAST': 1, 'PNPP_YEAST': 1,
                    'GR78_YEAST': 1, 'GALY_YEAST': 1, 'TCPD_YEAST': 1, 'DHSB_YEAST': 1, 'SSN6_YEAST': 1,
                    'SST2_YEAST': 1, 'AROG_YEAST': 1, '6PGD_YEAST': 1, 'R19A_YEAST': 1, 'PT94_YEAST': 1,
                    'VATD_YEAST': 1, 'PPB_YEAST ': 1, 'YKA5_YEAST': 1, 'UBC5_YEAST': 1, 'KHS1_YEAST': 1,
                    'PR39_YEAST': 1, 'YHA9_YEAST': 1, 'SAC7_YEAST': 1, 'ODPA_YEAST': 1, 'ACE2_YEAST': 1,
                    'RM37_YEAST': 1, 'ATR1_YEAST': 1, 'H2B1_YEAST': 1, 'CC27_YEAST': 1, 'TPS3_YEAST': 1,
                    'EXG1_YEAST': 1, 'RS11_YEAST': 1, 'CC25_YEAST': 1, 'CYBM_YEAST': 1, 'S61A_YEAST': 1,
                    'NI96_YEAST': 1, 'ERG6_YEAST': 1, 'YB72_YEAST': 1, 'UBP3_YEAST': 1, 'SIR3_YEAST': 1,
                    'SKO1_YEAST': 1, 'PTP2_YEAST': 1, 'FLO1_YEAST': 1, 'SC18_YEAST': 1, 'RL2A_YEAST': 1,
                    'YE14_YEAST': 1, 'LIP5_YEAST': 1, 'KC21_YEAST': 1, 'TBF1_YEAST': 1, 'RL3P_YEAST': 1,
                    'MCM3_YEAST': 1, 'RPBY_YEAST': 1, 'RL71_YEAST': 1, 'ATP8_YEAST': 1, 'HS76_YEAST': 1,
                    'CSE1_YEAST': 1, 'MDJ1_YEAST': 1, 'YCR8_YEAST': 1, 'CBF1_YEAST': 1, 'TNT_YEAST ': 1,
                    'DPOA_YEAST': 1, 'RT17_YEAST': 1, 'MS51_YEAST': 1, 'K6P2_YEAST': 1, 'CEM1_YEAST': 1,
                    'YHI7_YEAST': 1, 'CPT1_YEAST': 1, 'RED1_YEAST': 1, 'NOP2_YEAST': 1, 'SDH4_YEAST': 1,
                    'INO2_YEAST': 1, 'YCZ6_YEAST': 1, 'R102_YEAST': 1, 'PUT2_YEAST': 1, 'ATC4_YEAST': 1,
                    'SPT2_YEAST': 1, 'RCA1_YEAST': 1, 'APE2_YEAST': 1, 'RL3B_YEAST': 1, 'MAD1_YEAST': 1,
                    'TPS1_YEAST': 1, 'RU17_YEAST': 1, 'TPM1_YEAST': 1, 'YCG9_YEAST': 1, 'COXA_YEAST': 1,
                    'ADH1_YEAST': 1, 'HO_YEAST  ': 1, 'RLA1_YEAST': 1, 'YIN0_YEAST': 1, 'E2BD_YEAST': 1,
                    'EST1_YEAST': 1, 'CATT_YEAST': 1, 'TP20_YEAST': 1, 'UCR6_YEAST': 1, 'GDA1_YEAST': 1,
                    'YM38_YEAST': 1, 'SIN3_YEAST': 1, 'GBB_YEAST ': 1, 'GDI1_YEAST': 1, 'CYC2_YEAST': 1,
                    'RSU2_YEAST': 1, 'SRS2_YEAST': 1, 'YPT7_YEAST': 1, 'RT28_YEAST': 1, 'MGM1_YEAST': 1,
                    'NAM9_YEAST': 1, 'CNA1_YEAST': 1, 'TCPE_YEAST': 1, 'UBC1_YEAST': 1, 'RT08_YEAST': 1,
                    'PAN1_YEAST': 1, 'H2A2_YEAST': 1, 'ENO2_YEAST': 1, 'G3P3_YEAST': 1, 'VRP1_YEAST': 1,
                    'FUR1_YEAST': 1, 'MDL2_YEAST': 1, 'TPM2_YEAST': 1, 'SIR1_YEAST': 1, 'RPB1_YEAST': 1,
                    'RTPT_YEAST': 1, 'YAB8_YEAST': 1, 'CDC3_YEAST': 1, 'PORI_YEAST': 1, 'ATH1_YEAST': 1,
                    'ATPY_YEAST': 1, 'EGD1_YEAST': 1, 'CC42_YEAST': 1, 'HS83_YEAST': 1, 'ARF1_YEAST': 1,
                    'FKBP_YEAST': 1, 'MGMT_YEAST': 1, 'IXR1_YEAST': 1, 'OTC_YEAST ': 1, 'GRPE_YEAST': 1,
                    'PIR1_YEAST': 1, 'SC13_YEAST': 1, 'CB32_YEAST': 1, 'RM41_YEAST': 1, 'GSP1_YEAST': 1,
                    'SAG1_YEAST': 1, 'STL1_YEAST': 1, 'MCE1_YEAST': 1, 'UBIQ_YEAST': 1, 'RS33_YEAST': 1,
                    'HAL1_YEAST': 1, 'ATE1_YEAST': 1, 'KCC1_YEAST': 1, 'VATB_YEAST': 1, 'HXT5_YEAST': 1,
                    'ZNRP_YEAST': 1, 'RPC6_YEAST': 1, 'PUR3_YEAST': 1, 'POP2_YEAST': 1, 'MNN1_YEAST': 1,
                    'HAP2_YEAST': 1, 'HS72_YEAST': 1, 'SNF3_YEAST': 1, 'COX4_YEAST': 1, 'MLP1_YEAST': 1,
                    'SOF1_YEAST': 1, 'RRN6_YEAST': 1, 'ATU1_YEAST': 1, 'RM06_YEAST': 1, 'F26_YEAST ': 1,
                    'UCRQ_YEAST': 1, 'RPB3_YEAST': 1, 'SHM1_YEAST': 1, 'RER1_YEAST': 1, 'PYR1_YEAST': 1,
                    'MSH3_YEAST': 1, 'YD66_YEAST': 1, 'NAB2_YEAST': 1, 'NHPA_YEAST': 1, 'PAS3_YEAST': 1,
                    'CTK1_YEAST': 1, 'SEC9_YEAST': 1, 'RT13_YEAST': 1, 'PGM2_YEAST': 1, 'VATA_YEAST': 1,
                    'TRPF_YEAST': 1, 'SPB4_YEAST': 1, 'SED4_YEAST': 1, 'RSD1_YEAST': 1, 'YCQ7_YEAST': 1,
                    'RS31_YEAST': 1, 'PHR_YEAST ': 1, 'SYTC_YEAST': 1, 'PGK_YEAST ': 1, 'DLDH_YEAST': 1,
                    'ANC1_YEAST': 1, 'SPT3_YEAST': 1, 'SGE1_YEAST': 1, 'LYS2_YEAST': 1, 'YKH4_YEAST': 1,
                    'MNS1_YEAST': 1, 'KRE6_YEAST': 1, 'NAM7_YEAST': 1, 'R26A_YEAST': 1, 'RM25_YEAST': 1,
                    'HS30_YEAST': 1, 'IF41_YEAST': 1, 'K6P1_YEAST': 1, 'S61G_YEAST': 1, 'ALG1_YEAST': 1,
                    'CAPB_YEAST': 1, 'STF1_YEAST': 1, 'HIR2_YEAST': 1, 'AMPM_YEAST': 1, 'IMP2_YEAST': 1,
                    'KIN1_YEAST': 1, 'PEM1_YEAST': 1, 'YB8E_YEAST': 1, 'DHE4_YEAST': 1, 'YHB7_YEAST': 1,
                    'RM20_YEAST': 1, 'MK11_YEAST': 1, 'MA3T_YEAST': 1, 'YCV1_YEAST': 1, 'ERG2_YEAST': 1,
                    'DATI_YEAST': 1, 'UCR9_YEAST': 1, 'YJ51_YEAST': 1, 'MAC1_YEAST': 1, 'NOP3_YEAST': 1,
                    'YCK1_YEAST': 1, 'YHJ2_YEAST': 1, 'GLYM_YEAST': 1, 'THIK_YEAST': 1, 'COXX_YEAST': 1,
                    'NAB1_YEAST': 1, 'YN70_YEAST': 1, 'MPCP_YEAST': 1, 'STH1_YEAST': 1, 'RA18_YEAST': 1,
                    'YJ36_YEAST': 1, 'YCS7_YEAST': 1, 'MDM1_YEAST': 1, 'G6PI_YEAST': 1, 'SOK1_YEAST': 1,
                    'COQ1_YEAST': 1, 'ACT5_YEAST': 1, 'MFA2_YEAST': 1, 'MER1_YEAST': 1, 'ILVB_YEAST': 1,
                    'ATPD_YEAST': 1, 'SUI1_YEAST': 1, 'RPM2_YEAST': 1, 'SNF4_YEAST': 1, 'COX9_YEAST': 1,
                    'MRF1_YEAST': 1, 'CB33_YEAST': 1, 'PR38_YEAST': 1, 'RPA3_YEAST': 1, 'YK44_YEAST': 1,
                    'HXTY_YEAST': 1, 'ER24_YEAST': 1, 'UNG_YEAST ': 1, 'FRE1_YEAST': 1, 'FBRL_YEAST': 1,
                    'SMY1_YEAST': 1, 'YKN4_YEAST': 1, 'MAK3_YEAST': 1, 'SUV3_YEAST': 1, 'PLSC_YEAST': 1,
                    'INV1_YEAST': 1, 'MSS1_YEAST': 1, 'YHR0_YEAST': 1, 'NHPB_YEAST': 1, 'LYP1_YEAST': 1,
                    'PRCD_YEAST': 1, 'FET3_YEAST': 1, 'GLN3_YEAST': 1, 'PRI1_YEAST': 1, 'RS3A_YEAST': 1,
                    'MFA4_YEAST': 1, 'PYRD_YEAST': 1, 'RM02_YEAST': 1, 'CHI2_YEAST': 1, 'PROC_YEAST': 1,
                    'COQ3_YEAST': 1, 'R142_YEAST': 1, 'YEU2_YEAST': 1, 'MYS4_YEAST': 1, 'SCA1_YEAST': 1,
                    'ALG8_YEAST': 1, 'GCR2_YEAST': 1, 'PCNA_YEAST': 1, 'RPC3_YEAST': 1, 'MR11_YEAST': 1,
                    'CLH_YEAST ': 1, 'HR25_YEAST': 1, 'NHP2_YEAST': 1, 'SNM1_YEAST': 1, 'CAP_YEAST ': 1,
                    'GLRX_YEAST': 1, 'CC48_YEAST': 1, 'PDC2_YEAST': 1, 'PH84_YEAST': 1, 'RFC4_YEAST': 1,
                    'HCM1_YEAST': 1, 'TEC1_YEAST': 1, 'GAL4_YEAST': 1, 'CCPR_YEAST': 1, 'SYH_YEAST ': 1,
                    'R29B_YEAST': 1, 'KICH_YEAST': 1, 'VM11_YEAST': 1, 'YHP0_YEAST': 1, 'SPT7_YEAST': 1,
                    'ATH2_YEAST': 1, 'ATPG_YEAST': 1, 'BLH1_YEAST': 1, 'RFT1_YEAST': 1, 'SX19_YEAST': 1,
                    'DBR1_YEAST': 1, 'PRCH_YEAST': 1, 'YKE6_YEAST': 1, 'ADA3_YEAST': 1, 'YBE2_YEAST': 1,
                    'RFC1_YEAST': 1, 'ADH3_YEAST': 1, 'ACOX_YEAST': 1, 'SC59_YEAST': 1, 'ST14_YEAST': 1,
                    'MRS4_YEAST': 1, 'SEC1_YEAST': 1, 'IF51_YEAST': 1, 'SNC1_YEAST': 1, 'UGS2_YEAST': 1,
                    'SLA2_YEAST': 1, 'TRPG_YEAST': 1, 'PHO2_YEAST': 1, 'SRPR_YEAST': 1, 'TCPB_YEAST': 1,
                    'CHS1_YEAST': 1, 'NSP1_YEAST': 1, 'SIC1_YEAST': 1, 'SLY1_YEAST': 1, 'CTPT_YEAST': 1,
                    'PUB1_YEAST': 1, 'UCR2_YEAST': 1, 'PRC2_YEAST': 1, 'SC15_YEAST': 1, 'COAC_YEAST': 1,
                    'R167_YEAST': 1, 'PHO4_YEAST': 1, 'SEC6_YEAST': 1, 'GLYC_YEAST': 1, 'H2B2_YEAST': 1,
                    'MSH2_YEAST': 1, 'PRC3_YEAST': 1, 'CH10_YEAST': 1, 'RN15_YEAST': 1, 'STV1_YEAST': 1,
                    'FET4_YEAST': 1, 'RM31_YEAST': 1, 'GAP1_YEAST': 1, 'YJ43_YEAST': 1, 'YUR1_YEAST': 1,
                    'EPT1_YEAST': 1, 'TSL1_YEAST': 1, 'YHF0_YEAST': 1, 'GPDA_YEAST': 1, 'G6PD_YEAST': 1,
                    'SPT5_YEAST': 1, 'TOP1_YEAST': 1, 'RAD1_YEAST': 1, 'DA81_YEAST': 1, 'TCPZ_YEAST': 1,
                    'STE3_YEAST': 1, 'SNP2_YEAST': 1, 'SHR3_YEAST': 1, 'NUF1_YEAST': 1, 'CAN1_YEAST': 1,
                    'PMS1_YEAST': 1, 'E2BE_YEAST': 1, 'STE5_YEAST': 1, 'PYC2_YEAST': 1, 'SEC8_YEAST': 1,
                    'YBG6_YEAST': 1, 'VATC_YEAST': 1, 'PHD1_YEAST': 1, 'GCN5_YEAST': 1, 'PH80_YEAST': 1,
                    'RM38_YEAST': 1, 'KRE5_YEAST': 1, 'TSM1_YEAST': 1, 'IDHP_YEAST': 1, 'YHR8_YEAST': 1,
                    'PRC6_YEAST': 1, 'NUF2_YEAST': 1, 'SWI4_YEAST': 1, 'CYPD_YEAST': 1, 'CC31_YEAST': 1,
                    'MAT1_YEAST': 1, 'CIN1_YEAST': 1, 'PTR2_YEAST': 1, 'SRB4_YEAST': 1, 'CC68_YEAST': 1,
                    'ATPS_YEAST': 1, 'SYN_YEAST ': 1, 'AP19_YEAST': 1, 'ARGD_YEAST': 1, 'SRP2_YEAST': 1,
                    'UGA4_YEAST': 1, 'R19B_YEAST': 1, 'HEM2_YEAST': 1, 'S120_YEAST': 1, 'HPR1_YEAST': 1,
                    'ATPU_YEAST': 1, 'SLY4_YEAST': 1, 'EFG1_YEAST': 1, 'IF5_YEAST ': 1, 'TYR1_YEAST': 1,
                    'MANA_YEAST': 1, 'YOX1_YEAST': 1, 'VP17_YEAST': 1, 'ADP1_YEAST': 1, 'SSD1_YEAST': 1,
                    'DUR3_YEAST': 1, 'RA25_YEAST': 1, 'GPT_YEAST ': 1, 'RAM2_YEAST': 1, 'PDI_YEAST ': 1,
                    'AMYH_YEAST': 1, 'PRC8_YEAST': 1, 'EFG2_YEAST': 1, 'RS3B_YEAST': 1, 'ODO2_YEAST': 1,
                    'ATPT_YEAST': 1, 'GEF1_YEAST': 1, 'IRA2_YEAST': 1, 'MIF2_YEAST': 1, 'R14B_YEAST': 1,
                    'CYPH_YEAST': 1, 'UGA3_YEAST': 1, 'TFS2_YEAST': 1, 'GLO3_YEAST': 1, 'ATP7_YEAST': 1,
                    'PUT4_YEAST': 1, 'YKW2_YEAST': 1, 'CARB_YEAST': 1, 'RCS1_YEAST': 1, 'GAS1_YEAST': 1,
                    'SMC1_YEAST': 1, 'SC65_YEAST': 1, 'CYC1_YEAST': 1, 'OAT_YEAST ': 1, 'MCM1_YEAST': 1,
                    'ABF2_YEAST': 1, 'COX7_YEAST': 1, 'YHY1_YEAST': 1, 'PPA1_YEAST': 1, 'ESP1_YEAST': 1,
                    'MCM2_YEAST': 1, 'ODPX_YEAST': 1, 'TOP2_YEAST': 1, 'E2BG_YEAST': 1, 'HXT1_YEAST': 1,
                    'CCHL_YEAST': 1, 'BDF1_YEAST': 1, 'HS60_YEAST': 1, 'SMI1_YEAST': 1, 'TRPD_YEAST': 1,
                    'CISZ_YEAST': 1, 'MPI2_YEAST': 1, 'AAR2_YEAST': 1, 'UBA1_YEAST': 1, 'R17A_YEAST': 1,
                    'MASY_YEAST': 1, 'DNLI_YEAST': 1, 'ENO1_YEAST': 1, 'CC6_YEAST ': 1, 'GALX_YEAST': 1,
                    'LCB2_YEAST': 1, 'NMT_YEAST ': 1, 'SS10_YEAST': 1, 'SUP2_YEAST': 1, 'R114_YEAST': 1,
                    'UBCX_YEAST': 1, 'SP21_YEAST': 1, 'GCN4_YEAST': 1, 'PT27_YEAST': 1, 'NSR1_YEAST': 1,
                    'PEP8_YEAST': 1, 'MIG1_YEAST': 1, 'SYI_YEAST ': 1, 'CBP6_YEAST': 1, 'YKA8_YEAST': 1,
                    'FKB3_YEAST': 1, 'SYV_YEAST ': 1, 'YBC6_YEAST': 1, 'S160_YEAST': 1, 'MSH1_YEAST': 1,
                    'HS82_YEAST': 1, 'MTD1_YEAST': 1, 'RIF1_YEAST': 1, 'ADT3_YEAST': 1, 'YN46_YEAST': 1,
                    'UCR8_YEAST': 1, 'ARO1_YEAST': 1, 'ERG7_YEAST': 1, 'YHC6_YEAST': 1, 'KTR1_YEAST': 1,
                    'SNF1_YEAST': 1, 'R271_YEAST': 1, 'YK68_YEAST': 1, 'RN12_YEAST': 1, 'YMC2_YEAST': 1,
                    'CBP3_YEAST': 1, 'YKY8_YEAST': 1, 'AGA2_YEAST': 1, 'CATA_YEAST': 1, 'RPBX_YEAST': 1,
                    'RFC3_YEAST': 1, 'SCJ1_YEAST': 1, 'RPCX_YEAST': 1, 'CHS2_YEAST': 1, 'TF2D_YEAST': 1,
                    'R37B_YEAST': 1, 'YIB3_YEAST': 1, 'AST1_YEAST': 1, 'YB8I_YEAST': 1, 'MYS2_YEAST': 1,
                    'REB1_YEAST': 1, 'IRE1_YEAST': 1, 'YCR3_YEAST': 1, 'SED1_YEAST': 1, 'SP11_YEAST': 1,
                    'RIM1_YEAST': 1, 'SRPI_YEAST': 1, 'YDA2_YEAST': 1, 'TPS2_YEAST': 1, 'TBA3_YEAST': 1,
                    'PYC1_YEAST': 1, 'VAC1_YEAST': 1, 'RL34_YEAST': 1, 'UBC7_YEAST': 1, 'RAD4_YEAST': 1,
                    'CHI1_YEAST': 1, 'LEU1_YEAST': 1, 'CAO_YEAST ': 1, 'TF3A_YEAST': 1, 'TF2B_YEAST': 1,
                    'ERG1_YEAST': 1, 'TF3B_YEAST': 1, 'RPB5_YEAST': 1, 'MRS1_YEAST': 1, 'SPA2_YEAST': 1,
                    'COXZ_YEAST': 1, 'PDR1_YEAST': 1, 'SMD1_YEAST': 1, 'CAP2_YEAST': 1, 'LAG1_YEAST': 1,
                    'UME5_YEAST': 1, 'SC21_YEAST': 1, 'SON1_YEAST': 1, 'OPI1_YEAST': 1, 'ACO1_YEAST': 1,
                    'SWI5_YEAST': 1, 'PRTB_YEAST': 1, 'SP14_YEAST': 1, 'SPT6_YEAST': 1, 'YJ91_YEAST': 1,
                    'ACR1_YEAST': 1, 'BCK1_YEAST': 1, 'GAL8_YEAST': 1, 'YGP1_YEAST': 1, 'YP51_YEAST': 1,
                    'TBB_YEAST ': 1, 'ATPA_YEAST': 1, 'ACH1_YEAST': 1, 'CBP4_YEAST': 1, 'CBF5_YEAST': 1,
                    'INO4_YEAST': 1, 'STE2_YEAST': 1, 'BEM1_YEAST': 1, 'NC5R_YEAST': 1, 'RF1M_YEAST': 1,
                    'KI28_YEAST': 1, 'RU1A_YEAST': 1, 'PUT3_YEAST': 1, 'YKJ5_YEAST': 1, 'GCN2_YEAST': 1,
                    'YKE9_YEAST': 1, 'HSF_YEAST ': 1, 'GCR1_YEAST': 1, 'ATM1_YEAST': 1, 'YK62_YEAST': 1,
                    'YMC1_YEAST': 1, 'MSN1_YEAST': 1, 'STP1_YEAST': 1, 'SPT4_YEAST': 1, 'SRB5_YEAST': 1,
                    'TRP_YEAST ': 1, 'ERD2_YEAST': 1, 'RM27_YEAST': 1, 'HXT2_YEAST': 1, 'MDHP_YEAST': 1,
                    'RGM1_YEAST': 1, 'RAD9_YEAST': 1, 'PEP3_YEAST': 1, 'VP16_YEAST': 1, 'RFA3_YEAST': 1,
                    'COX1_YEAST': 1, 'MDHC_YEAST': 1, 'SWI6_YEAST': 1, 'KRE2_YEAST': 1, 'CIK1_YEAST': 1,
                    'EF3_YEAST ': 1, 'PRTD_YEAST': 1, 'RPC8_YEAST': 1, 'R16A_YEAST': 1, 'LCF2_YEAST': 1,
                    'NMD2_YEAST': 1, 'ITR2_YEAST': 1, 'HBS1_YEAST': 1, 'GLNA_YEAST': 1, 'TSA_YEAST ': 1,
                    'EUG1_YEAST': 1, 'MDL1_YEAST': 1, 'SYDC_YEAST': 1, 'TCPA_YEAST': 1, 'CBP2_YEAST': 1,
                    'LYS1_YEAST': 1, 'TRM1_YEAST': 1, 'RL3E_YEAST': 1, 'SYWM_YEAST': 1, 'RPC4_YEAST': 1,
                    'SC72_YEAST': 1, 'LONM_YEAST': 1, 'ACT_YEAST ': 1, 'GAL1_YEAST': 1, 'PRC9_YEAST': 1,
                    'RL1_YEAST ': 1, 'CYT2_YEAST': 1, 'DBP2_YEAST': 1, 'KRE1_YEAST': 1, 'END3_YEAST': 1,
                    'CHMU_YEAST': 1, 'HXKG_YEAST': 1, 'CISY_YEAST': 1, 'YKM9_YEAST': 1, 'MA3R_YEAST': 1,
                    'SC66_YEAST': 1, 'CBP1_YEAST': 1, 'RA50_YEAST': 1, 'APE3_YEAST': 1, 'VAL1_YEAST': 1,
                    'SMF2_YEAST': 1, 'PDR3_YEAST': 1, 'YHA8_YEAST': 1, 'UBP2_YEAST': 1, 'SC20_YEAST': 1,
                    'NIP1_YEAST': 1, 'PT12_YEAST': 1, 'EF2_YEAST ': 1, 'MT28_YEAST': 1, 'RA10_YEAST': 1,
                    'LEU3_YEAST': 1, 'TYSY_YEAST': 1, 'KEX2_YEAST': 1, 'BET1_YEAST': 1, 'YJ13_YEAST': 1,
                    'SYQ_YEAST ': 1, 'DCP1_YEAST': 1, 'ADT1_YEAST': 1, 'MS18_YEAST': 1, 'INV4_YEAST': 1,
                    'AFG3_YEAST': 1, 'PH85_YEAST': 1, 'RM08_YEAST': 1, 'YAE2_YEAST': 1, 'NAP1_YEAST': 1,
                    'RL27_YEAST': 1, 'CCL1_YEAST': 1, 'NOT4_YEAST': 1, 'CARP_YEAST': 1, 'IMP1_YEAST': 1,
                    'DBP1_YEAST': 1, 'DCP3_YEAST': 1, 'ST12_YEAST': 1, 'HDF1_YEAST': 1, 'ODPB_YEAST': 1,
                    'ADH4_YEAST': 1, 'RAD2_YEAST': 1, 'PAS1_YEAST': 1, 'SNF6_YEAST': 1, 'YCS4_YEAST': 1,
                    'DPB3_YEAST': 1, 'NOT3_YEAST': 1, 'KAPB_YEAST': 1, 'RLA2_YEAST': 1, 'SYMM_YEAST': 1,
                    'SMY2_YEAST': 1, 'RL9_YEAST ': 1, 'GLGB_YEAST': 1, 'RPB4_YEAST': 1, 'RM33_YEAST': 1,
                    'USO1_YEAST': 1, 'MPI1_YEAST': 1, 'ACEA_YEAST': 1, 'CG11_YEAST': 1, 'RA14_YEAST': 1,
                    'PMGY_YEAST': 1, 'TREA_YEAST': 1, 'STF2_YEAST': 1, 'UBC6_YEAST': 1, 'ITR1_YEAST': 1,
                    'KAD1_YEAST': 1, 'JNM1_YEAST': 1, 'RM09_YEAST': 1, 'SYG_YEAST ': 1, 'RAS2_YEAST': 1,
                    'SEN2_YEAST': 1, 'BIK1_YEAST': 1, 'SEC4_YEAST': 1, 'CALX_YEAST': 1, 'ALP1_YEAST': 1,
                    'PR04_YEAST': 1, 'DYR_YEAST ': 1, 'MNN9_YEAST': 1, 'AROF_YEAST': 1, 'GLS1_YEAST': 1,
                    'RM36_YEAST': 1, 'KIME_YEAST': 1, 'RM04_YEAST': 1, 'COXB_YEAST': 1, 'PRC5_YEAST': 1,
                    'PHSG_YEAST': 1, 'HEM6_YEAST': 1, 'PR22_YEAST': 1, 'SKI3_YEAST': 1, 'YCY8_YEAST': 1,
                    'BUD5_YEAST': 1, 'HMD2_YEAST': 1, 'EF1B_YEAST': 1, 'CYPC_YEAST': 1, 'KAPC_YEAST': 1,
                    'AP54_YEAST': 1, 'RL46_YEAST': 1, 'CLC1_YEAST': 1, 'SYFB_YEAST': 1, 'AMYG_YEAST': 1,
                    'LGT3_YEAST': 1, 'OCH1_YEAST': 1, 'ACP_YEAST ': 1, 'RPB8_YEAST': 1, 'SODC_YEAST': 1,
                    'TOA2_YEAST': 1, 'CYB_YEAST ': 1, 'SC22_YEAST': 1, 'G3P2_YEAST': 1, 'YHE0_YEAST': 1,
                    'CNA2_YEAST': 1, 'ADB1_YEAST': 1, 'RAT1_YEAST': 1, 'ATP6_YEAST': 1, 'NUC1_YEAST': 1,
                    'OSTD_YEAST': 1, 'RPC9_YEAST': 1, 'DPB2_YEAST': 1, 'SYKM_YEAST': 1, 'SX18_YEAST': 1,
                    'ATPB_YEAST': 1, 'SRB6_YEAST': 1, 'IF2B_YEAST': 1, 'R29A_YEAST': 1, 'SIP2_YEAST': 1,
                    'RL3_YEAST ': 1, 'SUG1_YEAST': 1, 'CYS3_YEAST': 1, 'GPDM_YEAST': 1, 'ZUO1_YEAST': 1,
                    'YEP0_YEAST': 1, 'FOX2_YEAST': 1, 'NIN1_YEAST': 1, 'RPB6_YEAST': 1, 'PMP1_YEAST': 1,
                    'PGM1_YEAST': 1, 'YIR3_YEAST': 1, 'G3P1_YEAST': 1, 'FUMH_YEAST': 1, 'SPK1_YEAST': 1,
                    'YB30_YEAST': 1, 'CACP_YEAST': 1, 'ARF2_YEAST': 1, 'RPC5_YEAST': 1, 'BOS1_YEAST': 1,
                    'PPR1_YEAST': 1, 'SNC2_YEAST': 1, 'RM32_YEAST': 1, 'HXT6_YEAST': 1, 'YAF3_YEAST': 1,
                    'ATC3_YEAST': 1, 'MTF1_YEAST': 1, 'GBA1_YEAST': 1, 'ABC1_YEAST': 1, 'DPOG_YEAST': 1,
                    'SC25_YEAST': 1, 'SYLC_YEAST': 1, 'ROX1_YEAST': 1, 'DHE2_YEAST': 1, 'YB32_YEAST': 1,
                    'LY14_YEAST': 1, 'OSTB_YEAST': 1, 'YJ49_YEAST': 1, 'SIS2_YEAST': 1, 'OM70_YEAST': 1,
                    'TRPE_YEAST': 1, 'PRCZ_YEAST': 1, 'NCPR_YEAST': 1, 'SIN4_YEAST': 1, 'CYC7_YEAST': 1,
                    'PAS7_YEAST': 1, 'SSO1_YEAST': 1, 'PIK1_YEAST': 1, 'YMX1_YEAST': 1, 'GCR3_YEAST': 1,
                    'ARG2_YEAST': 1, 'ODO1_YEAST': 1, 'KRE9_YEAST': 1, 'RHO1_YEAST': 1, 'UME6_YEAST': 1,
                    'RT02_YEAST': 1, 'RMAR_YEAST': 1, 'RNA1_YEAST': 1, 'COX3_YEAST': 1, 'MYS1_YEAST': 1,
                    'ATC1_YEAST': 1, 'DCP2_YEAST': 1, 'IF4E_YEAST': 1, 'RS37_YEAST': 1, 'RAD5_YEAST': 1,
                    'ILV5_YEAST': 1, 'SYG1_YEAST': 1, 'YHY0_YEAST': 1, 'VP34_YEAST': 1, 'CYPB_YEAST': 1,
                    'UBC3_YEAST': 1, 'HOP1_YEAST': 1, 'C1TM_YEAST': 1, 'YEA6_YEAST': 1, 'GAR1_YEAST': 1,
                    'AGA1_YEAST': 1, 'COXE_YEAST': 1, 'DA80_YEAST': 1, 'MET2_YEAST': 1, 'COQ2_YEAST': 1,
                    'RPB2_YEAST': 1, 'YHX8_YEAST': 1, 'MTA1_YEAST': 1, 'TFC1_YEAST': 1, 'ADR1_YEAST': 1,
                    'ST20_YEAST': 1, 'CALM_YEAST': 1, 'YAC2_YEAST': 1, 'CC23_YEAST': 1, 'PEP1_YEAST': 1,
                    'GBP2_YEAST': 1, 'NAM1_YEAST': 1, 'ZIP1_YEAST': 1, 'LEUR_YEAST': 1, 'YP53_YEAST': 1,
                    'COX6_YEAST': 1, 'RAD7_YEAST': 1, 'SIR4_YEAST': 1, 'SP10_YEAST': 1, 'SNF2_YEAST': 1,
                    'COFI_YEAST': 1, 'NPL1_YEAST': 1, 'KTR4_YEAST': 1, 'ATPO_YEAST': 1, 'LEU2_YEAST': 1,
                    'HXT3_YEAST': 1, 'MS16_YEAST': 1, 'END2_YEAST': 1, 'THRC_YEAST': 1, 'AROC_YEAST': 1,
                    'DPOD_YEAST': 1, 'CBS1_YEAST': 1, 'N100_YEAST': 1, 'YKS8_YEAST': 1, 'CBS2_YEAST': 1,
                    'RM16_YEAST': 1, 'YIG3_YEAST': 1, 'UBC4_YEAST': 1, 'AATC_YEAST': 1, 'YHB9_YEAST': 1,
                    'HS26_YEAST': 1, 'KCC2_YEAST': 1, 'YAB1_YEAST': 1, 'RPB9_YEAST': 1, 'PMT_YEAST ': 1,
                    'CYB2_YEAST': 1, 'SUL1_YEAST': 1, 'SYFM_YEAST': 1, 'SLN1_YEAST': 1, 'RFC2_YEAST': 1,
                    'SRB2_YEAST': 1, 'SNF5_YEAST': 1, 'RHO2_YEAST': 1, 'FIMB_YEAST': 1, 'DAP1_YEAST': 1,
                    'YHT3_YEAST': 1, 'RA51_YEAST': 1, 'RCC_YEAST ': 1, 'MOT1_YEAST': 1, 'FUS3_YEAST': 1,
                    'KAD2_YEAST': 1, 'ERG8_YEAST': 1, 'YN19_YEAST': 1, 'AATM_YEAST': 1, 'RLA3_YEAST': 1,
                    'RL16_YEAST': 1, 'PIF1_YEAST': 1, 'PT17_YEAST': 1, 'KHR1_YEAST': 1, 'PT09_YEAST': 1,
                    'C1TC_YEAST': 1, 'ADH2_YEAST': 1, 'VP15_YEAST': 1, 'MK16_YEAST': 1, 'DHA1_YEAST': 1,
                    'HIS7_YEAST': 1, 'IF2M_YEAST': 1, 'ATC2_YEAST': 1, 'YHG2_YEAST': 1, 'VATE_YEAST': 1,
                    'PR19_YEAST': 1, 'RIM2_YEAST': 1, 'YJ16_YEAST': 1, 'RA55_YEAST': 1, 'RME1_YEAST': 1,
                    'VATX_YEAST': 1, 'YCA9_YEAST': 1, 'YIN4_YEAST': 1, 'RT01_YEAST': 1, 'OM45_YEAST': 1,
                    'KAR3_YEAST': 1, 'YHN8_YEAST': 1, 'CTR1_YEAST': 1, 'YCD8_YEAST': 1, 'PT11_YEAST': 1,
                    'END1_YEAST': 1, 'IPPI_YEAST': 1, 'RL13_YEAST': 1, 'CC12_YEAST': 1, 'RA52_YEAST': 1,
                    'RPA9_YEAST': 1, 'ATN1_YEAST': 1, 'RPD3_YEAST': 1, 'R141_YEAST': 1, 'INV2_YEAST': 1,
                    'THIL_YEAST': 1, 'KIP1_YEAST': 1, 'MA6T_YEAST': 1, 'VATF_YEAST': 1, 'FUR4_YEAST': 1,
                    'TRK1_YEAST': 1, 'KC22_YEAST': 1, 'RS3_YEAST ': 1, 'PTSR_YEAST': 1, 'CAT8_YEAST': 1,
                    'TKT2_YEAST': 1, 'ERG4_YEAST': 1, 'CALB_YEAST': 1, 'ACON_YEAST': 1, 'RN14_YEAST': 1,
                    'IDH2_YEAST': 1, 'UBP1_YEAST': 1, 'MPP2_YEAST': 1, 'IATP_YEAST': 1, 'IF42_YEAST': 1,
                    'HXKB_YEAST': 1, 'SIR2_YEAST': 1, 'PRI2_YEAST': 1, 'AP17_YEAST': 1, 'MDHM_YEAST': 1,
                    'CBPS_YEAST': 1, 'PUT1_YEAST': 1, 'SODM_YEAST': 1, 'CSG2_YEAST': 1, 'IF1A_YEAST': 1,
                    'IRA1_YEAST': 1, 'NDC1_YEAST': 1, 'NUP2_YEAST': 1, 'NUP1_YEAST': 1, 'PPA5_YEAST': 1,
                    'IF2A_YEAST': 1, 'CC24_YEAST': 1, 'KAPR_YEAST': 1, 'PPA3_YEAST': 1, 'YCFI_YEAST': 1,
                    'EFTU_YEAST': 1, 'OM20_YEAST': 1, 'PABP_YEAST': 1, 'SYKC_YEAST': 1, 'MSN4_YEAST': 1,
                    'TRAM_YEAST': 1, 'PIS_YEAST ': 1, 'MAS6_YEAST': 1, 'KC2C_YEAST': 1, 'PTM1_YEAST': 1,
                    'IPYR_YEAST': 1, 'SSO2_YEAST': 1, 'PLB1_YEAST': 1, 'KTHY_YEAST': 1, 'RL6_YEAST ': 1,
                    'ALF_YEAST ': 1, 'HXT4_YEAST': 1, 'CHS3_YEAST': 1, 'DPOE_YEAST': 1, 'PRCF_YEAST': 1,
                    'ERG3_YEAST': 1, 'SFL1_YEAST': 1, 'MPP1_YEAST': 1, 'RAM1_YEAST': 1, 'RFA1_YEAST': 1,
                    'YBI8_YEAST': 1, 'P2B1_YEAST': 1, 'ASG2_YEAST': 1, 'FPS1_YEAST': 1, 'RM49_YEAST': 1,
                    'TIF3_YEAST': 1, 'SDH3_YEAST': 1, 'COX2_YEAST': 1, 'H150_YEAST': 1, 'GOG5_YEAST': 1,
                    'YPT1_YEAST': 1, 'MEP1_YEAST': 1, 'SRP1_YEAST': 1, 'MD10_YEAST': 1, 'DLD1_YEAST': 1,
                    'HXT7_YEAST': 1, 'RRN7_YEAST': 1, 'SLP1_YEAST': 1, 'RL4A_YEAST': 1, 'PDR4_YEAST': 1,
                    'RA26_YEAST': 1, 'FPPS_YEAST': 1, 'KSS1_YEAST': 1, 'RAD3_YEAST': 1, 'AGAL_YEAST': 1,
                    'MER2_YEAST': 1, 'YHD5_YEAST': 1, 'RPA1_YEAST': 1, 'PPAB_YEAST': 1, 'TOA1_YEAST': 1,
                    'KIP2_YEAST': 1, 'SC62_YEAST': 1, 'TAT2_YEAST': 1, 'SMF1_YEAST': 1, 'UCRI_YEAST': 1,
                    'FHL1_YEAST': 1, 'ATPE_YEAST': 1, 'POB1_YEAST': 1, 'MSF1_YEAST': 1, 'MSB2_YEAST': 1,
                    'UCRX_YEAST': 1, 'YCQ0_YEAST': 1, 'RS15_YEAST': 1, 'YB8G_YEAST': 1, 'AFR1_YEAST': 1,
                    'VP45_YEAST': 1, 'PR28_YEAST': 1, 'KAR1_YEAST': 1, 'CY1_YEAST ': 1, 'YB91_YEAST': 1,
                    'PR16_YEAST': 1, 'NU49_YEAST': 1, '6P2K_YEAST': 1, 'BAR1_YEAST': 1, 'PR02_YEAST': 1,
                    'PMT1_YEAST': 1, 'DCUP_YEAST': 1, 'ASSY_YEAST': 1, 'RL4B_YEAST': 1, 'IF52_YEAST': 1,
                    'INO1_YEAST': 1, 'SEC2_YEAST': 1, 'ADB2_YEAST': 1, 'YHK8_YEAST': 1, 'HAP3_YEAST': 1,
                    'HMD1_YEAST': 1, 'RFA2_YEAST': 1, 'PMP2_YEAST': 1, 'KTR2_YEAST': 1, 'ROX3_YEAST': 1,
                    'COX8_YEAST': 1, 'RA23_YEAST': 1, 'SYTM_YEAST': 1, 'SYDM_YEAST': 1, 'HS73_YEAST': 1,
                    'LOS1_YEAST': 1, 'UCR1_YEAST': 1, 'SDHL_YEAST': 1, 'YHU2_YEAST': 1, 'YCK2_YEAST': 1,
                    'FAS1_YEAST': 1, 'MBP1_YEAST': 1, 'SYR_YEAST ': 1, 'FAD1_YEAST': 1, 'PR06_YEAST': 1,
                    'FKB2_YEAST': 1, 'PR05_YEAST': 1, 'CARA_YEAST': 1, 'VPH1_YEAST': 1, 'DRS1_YEAST': 1,
                    'CG13_YEAST': 1, 'DAL4_YEAST': 1, 'PDR5_YEAST': 1, 'ARG1_YEAST': 1, 'MRS2_YEAST': 1,
                    'YCX9_YEAST': 1, 'CACM_YEAST': 1, 'GTS1_YEAST': 1, 'H104_YEAST': 1, 'CC10_YEAST': 1,
                    'KEX1_YEAST': 1, 'PMM_YEAST ': 1, 'YKD8_YEAST': 1, 'PR21_YEAST': 1, 'RPB7_YEAST': 1,
                    'ARG3_YEAST': 1, 'ARLY_YEAST': 1, 'YKW0_YEAST': 1, 'TKT1_YEAST': 1, 'YAG7_YEAST': 1,
                    'GFA1_YEAST': 1, 'NDI1_YEAST': 1, 'SYLM_YEAST': 1, 'SYMC_YEAST': 1, 'T2EA_YEAST': 1,
                    'DPSD_YEAST': 1, 'COXG_YEAST': 1, 'YJ12_YEAST': 1, 'DAL5_YEAST': 1, 'YBR8_YEAST': 1,
                    'HEM1_YEAST': 1, 'ACE1_YEAST': 1, 'AR56_YEAST': 1, 'ALG5_YEAST': 1, 'TRK2_YEAST': 1,
                    'FZF1_YEAST': 1, 'RM13_YEAST': 1, 'SPR1_YEAST': 1, 'SP23_YEAST': 1, 'SC14_YEAST': 1,
                    'ABP1_YEAST': 1, 'HS71_YEAST': 1, 'PRCI_YEAST': 1, 'DMC1_YEAST': 1, 'GAL7_YEAST': 1,
                    'CHL1_YEAST': 1, 'MA6R_YEAST': 1, 'RPC1_YEAST': 1, 'SR72_YEAST': 1, 'TCTP_YEAST': 1,
                    'ATP9_YEAST': 1, 'RS21_YEAST': 1, 'HIR1_YEAST': 1, 'TYE7_YEAST': 1, 'YMC4_YEAST': 1,
                    'SEN1_YEAST': 1, 'DPO2_YEAST': 1, 'RIMI_YEAST': 1, 'SC12_YEAST': 1, 'MT17_YEAST': 1,
                    'CP51_YEAST': 1, 'SYFA_YEAST': 1, 'KAPA_YEAST': 1, 'IDH1_YEAST': 1, 'YAP3_YEAST': 1,
                    'CC16_YEAST': 1, 'CRM1_YEAST': 1, 'BCS1_YEAST': 1, 'UGS1_YEAST': 1, 'RL21_YEAST': 1,
                    'CIN4_YEAST': 1, 'NFS1_YEAST': 1, 'RPOM_YEAST': 1, 'PT91_YEAST': 1, 'GCS1_YEAST': 1,
                    'YAG3_YEAST': 1, 'PRT1_YEAST': 1, 'HS74_YEAST': 1, 'R61A_YEAST': 1, 'ISP6_YEAST': 1,
                    'RLA4_YEAST': 1, 'UMPK_YEAST': 1, 'NOT2_YEAST': 1, 'PTP1_YEAST': 1, 'CAPA_YEAST': 1,
                    'PIR3_YEAST': 1, 'DKA1_YEAST': 1, 'MAG_YEAST ': 1, 'YAB9_YEAST': 1, 'PR09_YEAST': 1,
                    'CCE1_YEAST': 1, 'YEX1_YEAST': 1, 'XRS2_YEAST': 1, 'DBF4_YEAST': 1, 'CC46_YEAST': 1,
                    'CYP1_YEAST': 1, 'GCN1_YEAST': 1, 'KIN2_YEAST': 1, 'MSP1_YEAST': 1, 'FUS1_YEAST': 1,
                    'VM12_YEAST': 1, 'R272_YEAST': 1, 'ATN2_YEAST': 1, 'ORC2_YEAST': 1, 'YJ90_YEAST': 1,
                    'SFP1_YEAST': 1, 'YBC4_YEAST': 1, 'CC4_YEAST ': 1, 'SRD1_YEAST': 1, 'YCH0_YEAST': 1,
                    'ANP1_YEAST': 1, 'SED5_YEAST': 1, 'YBY2_YEAST': 1, 'HIP1_YEAST': 1, 'YME1_YEAST': 1,
                    'IPY2_YEAST': 1, 'UBR1_YEAST': 1, 'BAS1_YEAST': 1, 'MAOX_YEAST': 1, 'P152_YEAST': 1,
                    'KPYK_YEAST': 1, 'YJ64_YEAST': 1, 'PSS_YEAST ': 1, 'SR68_YEAST': 1, 'HS77_YEAST': 1,
                    'TOR2_YEAST': 1, 'IPB2_YEAST': 1, 'PRCE_YEAST': 1, 'STE6_YEAST': 1, 'RA57_YEAST': 1,
                    'CC11_YEAST': 1, 'DHH1_YEAST': 1, 'VPS1_YEAST': 1, 'FCY2_YEAST': 1, 'APN1_YEAST': 1,
                    'PR08_YEAST': 1, 'FMT_YEAST ': 1, 'RLA0_YEAST': 1, 'TFB1_YEAST': 1, 'RPA2_YEAST': 1,
                    'DA82_YEAST': 1, 'TBA1_YEAST': 1, 'MET4_YEAST': 1, 'HSP7_YEAST': 1, 'SR14_YEAST': 1,
                    'IME1_YEAST': 1, 'TUP1_YEAST': 1, 'TOP3_YEAST': 1, 'TTP1_YEAST': 1, 'GAL2_YEAST': 1,
                    'RM44_YEAST': 1, 'DED1_YEAST': 1, 'H2A1_YEAST': 1, 'SLU7_YEAST': 1, 'YCW2_YEAST': 1,
                    'SUP1_YEAST': 1, 'TCPG_YEAST': 1, 'YNE2_YEAST': 1, 'CB31_YEAST': 1, 'PRCG_YEAST': 1,
                    'YB52_YEAST': 1, 'CSE2_YEAST': 1, 'CC40_YEAST': 1, 'LCB1_YEAST': 1, 'CAD1_YEAST': 1,
                    'DEP1_YEAST': 1, 'PT22_YEAST': 1, 'DSS4_YEAST': 1, 'R37A_YEAST': 1, 'STU1_YEAST': 1,
                    'YB54_YEAST': 1, 'PE12_YEAST': 1, 'HAP4_YEAST': 1, 'DAP2_YEAST': 1, 'MDS1_YEAST': 1,
                    'YP52_YEAST': 1, 'CBPY_YEAST': 1, 'TPIS_YEAST': 1, 'MSH4_YEAST': 1, 'BSD2_YEAST': 1,
                    'COT1_YEAST': 1, 'HS75_YEAST': 1, 'NOP4_YEAST': 1, 'DUN1_YEAST': 1, 'YIA6_YEAST': 1,
                    'YINO_YEAST': 1, 'MOD5_YEAST': 1, 'KTR3_YEAST': 1, 'YB48_YEAST': 1, 'CYAA_YEAST': 1,
                    'SPT8_YEAST': 1, 'FDFT_YEAST': 1, 'IF2G_YEAST': 1, 'HEX2_YEAST': 1, 'P2B2_YEAST': 1,
                    'F16P_YEAST': 1, 'SKN7_YEAST': 1, 'FAS2_YEAST': 1, 'RMR3_YEAST': 1, 'PROF_YEAST': 1,
                    'PDS1_YEAST': 1, 'PGD1_YEAST': 1, 'YK42_YEAST': 1, 'GAC1_YEAST': 1, 'BAF1_YEAST': 1,
                    'RL25_YEAST': 1, 'YIF2_YEAST': 1, 'RPC2_YEAST': 1, 'TIP1_YEAST': 1, 'PAP_YEAST ': 1,
                    'HEMZ_YEAST': 1, 'YAA7_YEAST': 1, 'ARGI_YEAST': 1, 'SSB1_YEAST': 1, 'ADR6_YEAST': 1,
                    'PT54_YEAST': 1, 'SAR1_YEAST': 1, 'DYHC_YEAST': 1, 'RT04_YEAST': 1, 'SMC2_YEAST': 1,
                    'AMPL_YEAST': 1, 'R17B_YEAST': 1, 'MA3S_YEAST': 1, 'SYAC_YEAST': 1, 'YB33_YEAST': 1,
                    'ODP2_YEAST': 1, 'EF1G_YEAST': 1, 'RA16_YEAST': 1, 'NAT1_YEAST': 1, 'PR18_YEAST': 1,
                    'CG12_YEAST': 1, 'IS42_YEAST': 1, 'HEM3_YEAST': 1, 'CAL1_YEAST': 1, 'SEC7_YEAST': 1,
                    'E2BB_YEAST': 1, 'GSP2_YEAST': 1, 'RT09_YEAST': 1, 'NAT2_YEAST': 1, 'YAF1_YEAST': 1,
                    'HXT8_YEAST': 1, 'SKN1_YEAST': 1, 'ERD1_YEAST': 1, 'RL17_YEAST': 1, 'YB37_YEAST': 1,
                    'DHA2_YEAST': 1, 'SYSC_YEAST': 1, 'PPCK_YEAST': 1, 'SIS1_YEAST': 1, 'MAN1_YEAST': 1,
                    'DPO4_YEAST': 1, 'MFA3_YEAST': 1, 'YBT6_YEAST': 1, 'GBG_YEAST ': 1, 'R16B_YEAST': 1,
                    'BCK2_YEAST': 1, 'METE_YEAST': 1, 'RS4_YEAST ': 1, 'MAS5_YEAST': 1, 'TRNL_YEAST': 1,
                    'PMT2_YEAST': 1, 'GCY_YEAST ': 1, 'CIN8_YEAST': 1, 'PP12_YEAST': 1, 'TFC4_YEAST': 1,
                    'MSN2_YEAST': 1, 'RAP1_YEAST': 1
                }
            },
            'mcg': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.11, 'max': 1.0, 'mean': 0.500121293800539, 'median': 0.49,
                    'std': 0.1372993003895818, 'skewness': 0.604291161361435, 'kurtosis': 0.4590599900673058
                }
            },
            'gvh': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.13, 'max': 1.0, 'mean': 0.4999326145552553, 'median': 0.49,
                    'std': 0.12392434900413846, 'skewness': 0.4166394134928298, 'kurtosis': 0.5561109616052562
                }
            },
            'alm': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.21, 'max': 1.0, 'mean': 0.500033692722372, 'median': 0.51,
                    'std': 0.08667024770783191, 'skewness': -0.22099540133881013, 'kurtosis': 1.60931957594235
                }
            },
            'mit': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0, 'mean': 0.26118598382749364, 'median': 0.22,
                    'std': 0.13709763089421498, 'skewness': 1.444776165204374, 'kurtosis': 2.2899705478348666
                }
            },
            'erl': {
                'var_type': 'discrete', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0.5: 1470, 1.0: 14
                    # 'min': 0.5, 'max': 1.0, 'mean': 0.5047169811320755, 'median': 0.5,
                    # 'std': 0.04835096692671292, 'skewness': 10.159632813962212, 'kurtosis': 101.35473389753923
                }
            },
            'pox': {
                'var_type': 'discrete', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0.00: 1469, 0.83: 11, 0.50: 4
                    # 'min': 0.0, 'max': 0.83, 'mean': 0.007500000000000001, 'median': 0.0,
                    # 'std': 0.0756826652050668, 'skewness': 10.276883707764107, 'kurtosis': 105.738712367983
                }
            },
            'vac': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.73, 'mean': 0.4998854447439337, 'median': 0.51,
                    'std': 0.05779658638925979, 'skewness': -1.7916406496353354, 'kurtosis': 9.501358534499472
                }
            },
            'nuc': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0, 'mean': 0.27619946091644665, 'median': 0.22,
                    'std': 0.10649052826089482, 'skewness': 2.41303121845126, 'kurtosis': 7.777761531779612
                }
            },
            'Localization_Site': {
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'CYT': 463, 'NUC': 429, 'MIT': 244, 'ME3': 163, 'ME2': 51,
                    'ME1': 44, 'EXC': 35, 'VAC': 30, 'POX': 20, 'ERL': 5
                }
            }
            ############################################################################################################
            # {  # SynSGAIN --> missing rate = 50%
            #     'Sequence_Name': {'6P2K_YEAST': 1484},
            #     'mcg': {
            #         'min': 0.46524568736553185, 'max': 0.5453022328950464,
            #         'mean': 0.5087020973143401, 'median': 0.5096154933795333,
            #         'std': 0.012544083214844333, 'skewness': -0.28693982273178076, 'kurtosis': -0.0816651412661904
            #     },
            #     'gvh': {
            #         'min': 0.46261394187808036, 'max': 0.5407283408008516,
            #         'mean': 0.5039414935061728, 'median': 0.5040366057306528,
            #         'std': 0.011906434630417989, 'skewness': -0.13040343730970413, 'kurtosis': 0.0282146709731812
            #     },
            #     'alm': {
            #         'min': 0.4565116715431214, 'max': 0.5327837663143874,
            #         'mean': 0.49367778610725693, 'median': 0.49369389695115384,
            #         'std': 0.009272332073867929, 'skewness': 0.049023738694154176, 'kurtosis': 0.879188349767444
            #     },
            #     'mit': {
            #         'min': 0.22323743999004364, 'max': 0.29204601466655733,
            #         'mean': 0.2528097558033596, 'median': 0.2528161223977804,
            #         'std': 0.01027158770723539, 'skewness': 0.17345292052707195, 'kurtosis': 0.1380184675716376
            #     },
            #     'erl': {0.5: 1484},
            #     'pox': {0.0: 1484},
            #     'vac': {
            #         'min': 0.4699452066421509, 'max': 0.530894358754158,
            #         'mean': 0.49911932979873447, 'median': 0.4994770030677318,
            #         'std': 0.009366391539524352, 'skewness': -0.10201703913073762, 'kurtosis': -0.1455023952761061
            #     },
            #     'nuc': {
            #         'min': 0.23307784587144845, 'max': 0.32336977884173385,
            #         'mean': 0.2746370675374631, 'median': 0.274297709017992,
            #         'std': 0.011466801629397333, 'skewness': 0.0634750185560705, 'kurtosis': 0.21900371564928367},
            #     'Localization_Site': {'CYT': 1484}
            # }
            ############################################################################################################
        }        # ok
    }
    """The metadata of the supported datasets.
    A dataset is identified by its (short) name (e.g., 'adult') which maps to the respective metadata of each column. 
    For the moment, the collected metadata of each column is relatively shallow and is composed by:
        - 'var_type': Either 'discrete' (aka categorical) or 'continuous' (aka numerical).
        - 'data_type': Such as int, float, str, bool, and so forth.
        - 'target': If True the column (i.e., the feature/variable) is the target variable. 
        - 'drop': If True the column (i.e., the feature/variable) is marked to be dropped.
        - 'missing_values': A list of symbols (e.g., [' ', 0, '?']) to be handled as missing values.
        - 'values_dist': If the column (i.e., the feature/variable) is 'discrete' then 
          the collected metadata is a dict of the values' occurrences. If the column is 'continuous' then 
          the collected metadata is a dict that stores the 'min', the 'max', the 'mean', the 'median', and 
          the 'std' (i.e., the standard deviation), the 'skewness', the 'kurtosis' values. 
          For other data types (e.g., str) it is still NOT defined the metadata that this dict (i.e., the 'values_dist') 
          should store.
    """

    @classmethod
    def discrete_vars(cls,
                      dataset: str = 'adult',
                      df: pd.DataFrame = None,
                      verbose: bool = False) -> Union[List[str], List[int]]:
        """Return a list of the discrete (aka categorical) variables (i.e., features/columns) of
        the given `dataset`. If the given pandas DataFrame (i.e., `df`) is NOT None then the list of its columns
        has to be a subset of the columns of the given `dataset` and
        the list to be returned has only the discrete columns that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars : Union[List[str], List[int]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_disc_vars: Union[List[str], List[int]] = [variable for variable, meta in Metadata.DATASETS[dataset].items()
                                                      if meta['var_type'] == 'discrete']
        disc_vars: Union[List[str], List[int]] = all_disc_vars

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: discrete_vars()\n"
                                 "The variables (i.e., features/columns) of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the ones of the given `dataset`.")
            disc_vars = [variable for variable in all_disc_vars if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: discrete_vars()")
            print(f"Discrete variables of the {dataset} dataset:\n{all_disc_vars}")
            print(f"Discrete variables to be returned:\n{disc_vars}")
        return disc_vars

    @classmethod
    def discrete_vars_and_dtypes(cls,
                                 dataset: str = 'Adult',
                                 df: pd.DataFrame = None,
                                 verbose: bool = False) -> Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]:
        """Return a list of the discrete (aka categorical) variables (i.e., features/columns) and
        their data types of the given `dataset`. If the given pandas DataFrame (i.e., `df`) is NOT None then
        the list of its columns has to be a subset of the columns of the given `dataset` and
        the list to be returned has only the discrete columns that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars_and_dtypes : Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) and
            their data types of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_disc_vars_and_dtypes: Union[List[Tuple[str, Any]], List[Tuple[int, Any]]] = [
            (variable, meta['data_type']) for variable, meta in Metadata.DATASETS[dataset].items()
            if meta['var_type'] == 'discrete']
        disc_vars_and_dtypes: Union[List[Tuple[str, Any]], List[Tuple[int, Any]]] = all_disc_vars_and_dtypes

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: discrete_vars_and_dtypes()\n"
                                 "The variables (i.e., features/columns) of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the ones of the given `dataset`.")
            disc_vars_and_dtypes = [(variable, data_type) for variable, data_type in all_disc_vars_and_dtypes
                                    if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: discrete_vars_and_dtypes()")
            print(f"Discrete variables and data types of the {dataset} dataset:\n{all_disc_vars_and_dtypes}")
            print(f"Discrete variables and data types to be returned:\n{disc_vars_and_dtypes}")
        return disc_vars_and_dtypes

    @classmethod
    def vars_to_drop(cls,
                     dataset: str = 'Adult',
                     df: pd.DataFrame = None,
                     verbose: bool = False) -> Union[List[str], List[int]]:
        """Return a list of the variables (i.e., features/columns) to be dropped of the given `dataset`.
        If the given pandas DataFrame (i.e., `df`) is NOT None then the list of its columns has to be
        a subset of the columns of the given `dataset` and
        the list to be returned has only variables that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars_and_dtypes : Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) and
            their data types of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_vars_to_drop: Union[List[str], List[int]] = [
            variable for variable, meta in Metadata.DATASETS[dataset].items() if meta['drop']]
        vars_to_drop: Union[List[str], List[int]] = all_vars_to_drop

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: vars_to_drop()\n"
                                 "The variables (i.e., features/columns) of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the ones of the given `dataset`.")
            vars_to_drop = [variable for variable in all_vars_to_drop if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: vars_to_drop()")
            print(f"Variables to drop of the {dataset} dataset:\n{all_vars_to_drop}")
            print(f"Variables to drop to be returned:\n{vars_to_drop}")
        return vars_to_drop

