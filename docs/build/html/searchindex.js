Search.setIndex({"docnames": ["api_reference", "generated/featuristic.datasets.fetch_cars_dataset", "generated/featuristic.selection.GeneticFeatureSelector", "generated/featuristic.synthesis.GeneticFeatureSynthesis", "generated/featuristic.synthesis.SymbolicFunction", "generated/featuristic.synthesis.list_operations", "guides/computational_performance", "guides/index", "guides/sklearn", "guides/tuning", "index", "install/index", "release_notes"], "filenames": ["api_reference.rst", "generated/featuristic.datasets.fetch_cars_dataset.rst", "generated/featuristic.selection.GeneticFeatureSelector.rst", "generated/featuristic.synthesis.GeneticFeatureSynthesis.rst", "generated/featuristic.synthesis.SymbolicFunction.rst", "generated/featuristic.synthesis.list_operations.rst", "guides/computational_performance.ipynb", "guides/index.rst", "guides/sklearn.ipynb", "guides/tuning.ipynb", "index.ipynb", "install/index.rst", "release_notes.rst"], "titles": ["API Reference", "featuristic.datasets.fetch_cars_dataset", "featuristic.selection.GeneticFeatureSelector", "featuristic.synthesis.GeneticFeatureSynthesis", "featuristic.synthesis.SymbolicFunction", "featuristic.synthesis.list_operations", "Computational Performance", "Guides", "Using Featuristic With scikit-learn Pipelines", "Tuning the Genetic Feature Synthesis", "What is Featuristic?", "Installation", "Release Notes"], "terms": {"sourc": [1, 2, 3, 4, 5], "fetch": 1, "car": [1, 8, 9, 10], "from": [1, 8, 9, 10, 11], "uci": 1, "repositori": [1, 11], "class": [2, 3, 4, 6, 8, 9, 10], "objective_funct": [2, 9, 10], "callabl": [2, 4], "population_s": [2, 3, 6, 8, 9, 10], "int": [2, 3, 4], "50": [2, 6, 8, 9, 10], "max_gener": [2, 3, 6, 8, 9, 10], "100": [2, 3, 6, 8, 9, 10], "crossover_proba": [2, 3], "float": [2, 3], "0": [2, 3, 6, 8, 9, 10], "9": [2, 8, 10], "mutation_proba": 2, "1": [2, 3, 6, 8, 9, 10], "early_termination_it": [2, 3, 6, 8, 9, 10], "15": [2, 3, 8, 10], "n_job": [2, 3, 6, 8, 9, 10], "pbar": [2, 3], "bool": [2, 3], "true": [2, 3], "verbos": [2, 3], "fals": [2, 3, 6, 9, 10], "The": [2, 3, 4, 5, 6, 8, 9], "genet": [2, 3, 6, 7, 8], "featur": [2, 3, 4, 6, 7, 8], "selector": [2, 10], "us": [2, 3, 4, 6, 7, 9, 10], "program": [2, 3, 6, 9], "best": [2, 3, 9, 10], "minimis": 2, "given": [2, 3, 9], "object": [2, 10], "function": [2, 3, 4], "thi": [2, 3, 6, 8, 9, 10], "i": [2, 3, 6, 8, 9, 11], "done": [2, 3], "initi": [2, 3, 4, 10], "build": [2, 3], "popul": [2, 3, 10], "naiv": [2, 3], "random": [2, 3, 6, 8, 9, 10], "avail": [2, 3, 5, 9, 10, 11], "evolv": [2, 3, 9, 10], "over": [2, 3, 10], "number": [2, 3, 4, 6, 9, 10], "gener": [2, 3, 4, 6, 8, 10], "oper": [2, 3, 5, 9, 10], "mutat": [2, 3, 9, 10], "crossov": [2, 3, 9], "find": [2, 3, 8, 9], "combin": [2, 8, 9, 10], "output": [2, 8, 10], "__init__": [2, 3, 4], "none": [2, 3, 10], "algorithm": [2, 3, 6, 8, 9, 10], "paramet": [2, 3, 4, 6, 9], "cost": [2, 6], "minim": [2, 3, 8, 10], "must": [2, 3], "take": [2, 3, 4, 6, 8, 9], "x": [2, 6, 8, 9, 10], "y": [2, 6, 8, 9, 10], "input": [2, 3], "return": [2, 3, 5, 8, 10], "note": [2, 10], "should": [2, 10], "valu": [2, 3, 6, 9, 10], "so": [2, 6, 10], "smaller": [2, 3, 6, 9, 10], "better": [2, 9, 10], "If": [2, 3, 6, 9, 10], "you": [2, 6, 8, 10], "want": [2, 10], "maxim": [2, 10], "metric": [2, 10], "multipli": [2, 10], "your": [2, 6, 8, 10], "individu": [2, 8, 9, 10], "maximum": [2, 3, 9], "iter": [2, 9, 10], "probabl": [2, 3], "wait": 2, "earli": [2, 3, 10], "termin": [2, 3], "parallel": [2, 3], "job": [2, 3], "run": [2, 3, 6, 11], "all": [2, 3, 10], "core": [2, 3], "els": [2, 3], "minimum": [2, 3], "cpu_count": 2, "whether": [2, 3, 9, 10], "print": [2, 3, 6, 8, 9, 10], "progress": [2, 3, 9], "method": [2, 3, 4], "fit": [3, 9, 10], "str": [3, 4, 5], "pearson": 3, "list": [3, 5], "symbolicfunct": 3, "num_featur": [3, 6, 8, 9, 10], "10": [3, 6, 8, 9, 10], "25": [3, 6, 8, 9, 10], "tournament_s": 3, "3": [3, 6, 8, 9, 10], "85": 3, "parsimony_coeffici": [3, 6, 8, 9, 10], "001": 3, "return_all_featur": [3, 6, 9, 10], "new": [3, 4, 6, 8, 9], "techniqu": 3, "base": 3, "symbol": [3, 4, 10], "regress": [3, 10], "formula": [3, 6, 8, 9, 10], "repres": [3, 4, 9], "transform": [3, 8, 10], "ar": [3, 6, 9, 10], "identifi": [3, 9, 10], "relev": 3, "redund": [3, 10], "mrmr": 3, "those": [3, 10], "most": [3, 9, 10], "correl": [3, 10], "target": [3, 10], "variabl": [3, 10], "while": [3, 10], "being": [3, 9], "least": 3, "each": [3, 9, 10], "other": [3, 8], "one": [3, 6, 10], "mae": [3, 10], "mse": 3, "spearman": 3, "built": [3, 5], "intern": [3, 4], "select": [3, 8, 9], "via": [3, 8, 10], "larger": [3, 6, 9], "more": [3, 6, 9, 10], "like": [3, 6, 9, 10], "good": 3, "solut": [3, 9], "longer": 3, "size": [3, 6], "tournament": 3, "between": [3, 9], "parent": 3, "parsimoni": 3, "coeffici": 3, "penal": [3, 6, 9], "encourag": [3, 6, 9], "help": [3, 9, 10], "prevent": [3, 9], "bloat": [3, 6, 9], "where": [3, 6, 9], "becom": [3, 6, 9], "increasingli": 3, "larg": [3, 6, 9], "complex": [3, 6, 8, 10], "without": [3, 6, 9], "improv": [3, 6, 9, 10], "which": [3, 6, 9, 10], "increas": [3, 6, 9], "comput": [3, 7, 9, 10], "reduc": [3, 6, 9, 10], "interpret": [3, 9, 10], "score": [3, 8, 10], "doe": 3, "just": [3, 6, 10], "serial": 3, "show": 3, "bar": 3, "out": [3, 10], "adit": 3, "inform": [3, 9], "func": 4, "arg_count": 4, "format_str": 4, "name": [4, 8, 10], "an": [4, 6, 9, 10], "geneticfeaturegener": 4, "argument": [4, 6, 9, 10], "format": 4, "string": 4, "type": 5, "There": [6, 9], "sever": [6, 9], "can": [6, 8, 9, 10], "featurist": [6, 7, 9, 11], "shown": [6, 10], "below": [6, 9, 10], "import": [6, 8, 9, 10], "ft": [6, 8, 9, 10], "numpi": [6, 8, 9, 10], "np": [6, 8, 9, 10], "seed": [6, 8, 9, 10], "8888": [6, 8, 9, 10], "__version__": [6, 8, 9, 10], "fetch_cars_dataset": [6, 8, 9, 10], "control": [6, 9], "mathemat": [6, 10], "express": [6, 10], "when": [6, 9, 10], "set": [6, 9, 10], "heavili": [6, 9], "therebi": [6, 9], "creation": [6, 9, 10], "excess": [6, 9], "By": [6, 8, 9, 10], "discourag": [6, 9], "overli": [6, 9], "calcul": 6, "quickli": 6, "In": [6, 9, 10], "exampl": [6, 8, 9, 10], "veri": [6, 8, 9], "small": [6, 9], "lead": [6, 9, 10], "time": 6, "2": [6, 8, 9, 10], "synth": [6, 9, 10], "geneticfeaturesynthesi": [6, 8, 9, 10], "5": [6, 8, 9, 10], "00001": [6, 9], "fit_transform": [6, 9, 10], "info": [6, 9, 10], "get_feature_info": [6, 8, 9, 10], "head": [6, 8, 9, 10], "iloc": [6, 9, 10], "creat": [6, 8, 9, 10], "58": [6, 9], "29": [6, 9], "00": [6, 8, 9, 10], "03": [6, 9], "lt": [6, 8, 9, 10], "6": [6, 8, 10], "83it": 6, "": [6, 8, 9, 10], "prune": [6, 8, 9, 10], "space": [6, 8, 9, 10], "679": 6, "06it": [6, 10], "02": [6, 9], "8": [6, 8, 9, 10], "73it": 6, "39": [6, 9, 10], "ab": [6, 8, 9, 10], "displac": [6, 8, 9, 10], "model_year": [6, 8, 9, 10], "weight": [6, 8, 9, 10], "sin": [6, 9, 10], "And": [6, 9, 10], "keep": [6, 9], "simpler": [6, 9], "mean": [6, 8, 9, 10], "thei": [6, 10], "60": [6, 9], "30": [6, 8, 9], "01": [6, 8, 9], "89it": 6, "601": 6, "75it": 6, "11": [6, 8, 9, 10], "88it": 6, "cube": [6, 9], "squar": [6, 8, 9, 10], "default": [6, 10], "geneticfeatureselector": [6, 8, 9, 10], "singl": 6, "cpu": [6, 10], "howev": [6, 9, 10], "nice": 6, "embarrassingli": 6, "both": [6, 10], "call": [6, 9, 10], "defin": [6, 8], "how": [6, 9, 10], "mani": 6, "spawn": 6, "continu": [6, 9, 10], "per": 6, "associ": 6, "datset": 6, "mai": [6, 9, 10], "actual": [6, 10], "effici": [6, 9, 10], "moder": 6, "dataset": [6, 8, 9, 10], "upward": 6, "see": [6, 10], "greater": 6, "than": 6, "It": [6, 9], "recommend": 6, "avoid": [6, 9], "significantli": 6, "machin": [6, 10], "resourc": [6, 9], "caus": [6, 10], "multi": 6, "slowli": 6, "4": [6, 8, 10], "With": [7, 10], "scikit": 7, "learn": [7, 10], "pipelin": 7, "tune": 7, "synthesi": 7, "perform": [7, 9, 10], "compat": 8, "power": 8, "allow": [8, 9], "organ": 8, "appli": [8, 10], "sequenc": 8, "process": [8, 10], "step": [8, 10], "effortlessli": 8, "within": [8, 10], "chain": 8, "togeth": 8, "variou": [8, 9], "provid": 8, "librari": 8, "These": [8, 10], "includ": 8, "scale": 8, "ani": [8, 9], "preprocess": 8, "requir": [8, 9, 10], "prepar": 8, "model": [8, 9], "leverag": 8, "conjunct": 8, "streamlin": 8, "workflow": 8, "ensur": [8, 10], "consist": [8, 10], "reproduc": 8, "construct": 8, "eas": 8, "develop": 8, "let": [8, 10], "look": [8, 10], "sklearn": [8, 10], "linear_model": [8, 10], "linearregress": [8, 10], "model_select": [8, 10], "cross_val_scor": [8, 10], "train_test_split": [8, 10], "cylind": [8, 10], "horsepow": [8, 10], "acceler": [8, 10], "origin": [8, 10], "307": [8, 10], "130": [8, 10], "3504": [8, 10], "12": [8, 10], "70": [8, 10], "350": [8, 10], "165": [8, 10], "3693": [8, 10], "318": [8, 10], "150": [8, 10], "3436": [8, 10], "304": [8, 10], "3433": [8, 10], "302": [8, 10], "140": [8, 10], "3449": [8, 10], "18": [8, 10], "16": [8, 10], "17": [8, 10], "mpg": [8, 10], "dtype": [8, 10], "float64": [8, 10], "x_train": [8, 10], "x_test": [8, 10], "y_train": [8, 10], "y_test": [8, 10], "test_siz": [8, 10], "we": [8, 9, 10], "custom": [8, 10], "pass": 8, "optim": [8, 9, 10], "subset": [8, 9, 10], "def": [8, 10], "cv": [8, 10], "neg_mean_absolute_error": [8, 10], "pipe": 8, "genetic_feature_synthesi": 8, "200": [8, 10], "genetic_feature_selector": 8, "28": [8, 10], "04": [8, 10], "87it": 8, "467": 8, "50it": 8, "84it": 8, "optimis": [8, 10], "56": [8, 10], "06": [8, 10], "05": [8, 10], "30it": 8, "7": [8, 9, 10], "pred": [8, 10], "predict": [8, 10], "arrai": 8, "44700795": 8, "35": 8, "29908224": 8, "26": 8, "6428252": 8, "54690644": 8, "27": [8, 10], "91071012": 8, "19": 8, "73099147": 8, "36": 8, "26329007": 8, "24": 8, "88007496": 8, "18340972": 8, "22": 8, "15965234": 8, "still": 8, "named_step": 8, "For": [8, 9, 10], "engin": [8, 10], "plot": 8, "histori": 8, "gf": 8, "feature_0": [8, 10], "co": 8, "911341": 8, "feature_1": [8, 10], "906715": 8, "feature_2": [8, 10], "904685": 8, "feature_3": [8, 10], "903028": 8, "feature_4": 8, "displa": 8, "896680": 8, "plot_histori": 8, "ll": [9, 10], "explor": 9, "enhanc": [9, 10], "21it": 9, "595": 9, "60it": 9, "91it": 9, "00it": 9, "607": 9, "39it": 9, "99it": 9, "refer": 9, "evolut": 9, "undergo": [9, 10], "befor": [9, 10], "cycl": 9, "candid": 9, "crucial": 9, "determin": 9, "durat": 9, "potenti": [9, 10], "onc": [9, 10], "specifi": 9, "reach": 9, "regardless": 9, "ha": [9, 10], "been": 9, "found": 9, "appropri": [9, 10], "balanc": 9, "abil": 9, "converg": 9, "satisfactori": 9, "too": 9, "low": 9, "prematur": 9, "convers": 9, "high": 9, "unnecessari": 9, "overhead": 9, "signific": 9, "qualiti": 9, "right": 9, "often": 9, "experiment": 9, "problem": 9, "specif": 9, "consider": 9, "desir": 9, "level": 9, "after": 9, "stop": 9, "hasn": 9, "t": 9, "manag": 9, "seem": 9, "have": [9, 10], "stall": 9, "assum": 9, "further": 9, "unlik": 9, "yield": [9, 10], "result": 9, "purpos": 9, "especi": 9, "deal": 9, "computation": 9, "expens": 9, "appear": 9, "work": 9, "well": [9, 10], "prolong": 9, "plateau": 9, "present": 9, "solv": 9, "influenc": 9, "divers": [9, 10], "capabl": [9, 10], "A": 9, "typic": 9, "broader": 9, "also": [9, 10], "slower": 9, "properli": 9, "faster": 9, "could": 9, "suffer": 9, "local": 9, "optima": 9, "due": 9, "limit": 9, "choos": 9, "act": 9, "exploit": 9, "speed": 9, "depend": [9, 11], "quantifi": [9, 10], "particular": 9, "respect": 9, "context": 9, "aim": 9, "gaug": 9, "evalu": [9, 10], "navig": 9, "toward": 9, "exhibit": [9, 10], "superior": 9, "term": 9, "preced": 9, "intens": 9, "task": 9, "hyperparamet": 9, "emploi": [9, 10], "straightforward": 9, "yet": 9, "effect": [9, 10], "rapid": 9, "identif": 9, "autom": 10, "intellig": 10, "deriv": 10, "fundament": 10, "add": 10, "subtract": 10, "tan": 10, "sqrt": 10, "instanc": 10, "might": 10, "next": 10, "assess": 10, "efficaci": 10, "quanifi": 10, "strongest": 10, "recombin": 10, "produc": 10, "offspr": 10, "illustr": 10, "point": 10, "alter": 10, "introduc": 10, "slight": 10, "variat": 10, "discoveri": 10, "novel": 10, "represent": 10, "across": 10, "multipl": 10, "refin": 10, "goal": 10, "strong": 10, "simpl": 10, "wide": 10, "two": 10, "distinct": 10, "phase": 10, "through": 10, "form": 10, "involv": 10, "design": 10, "captur": 10, "relationship": 10, "follow": [10, 11], "search": 10, "newli": 10, "accuraci": 10, "mean_absolute_error": 10, "now": 10, "dive": 10, "fun": 10, "part": 10, "automat": 10, "our": 10, "proce": 10, "robust": 10, "To": [10, 11], "achiev": 10, "first": 10, "split": 10, "test": 10, "remain": 10, "unseen": 10, "dure": 10, "serv": 10, "independ": 10, "ve": 10, "configur": 10, "synthes": 10, "u": 10, "entail": 10, "halt": 10, "fail": 10, "upon": 10, "addition": 10, "enabl": 10, "concurr": 10, "execut": 10, "everyth": 10, "up": 10, "simpli": 10, "0025": 10, "74it": 10, "622": 10, "14it": 10, "datafram": 10, "contain": 10, "synthesis": 10, "instead": 10, "generated_featur": 10, "feature_7": 10, "feature_8": 10, "122": 10, "86": 10, "2220": 10, "14": 10, "71": 10, "072622": 10, "049748e": 10, "09": 10, "206072": 10, "467213": 10, "966114e": 10, "210512": 10, "88": 10, "3060": 10, "81": 10, "056756": 10, "411911e": 10, "212173": 10, "605000": 10, "596968e": 10, "218293": 10, "129": 10, "3725": 10, "13": 10, "79": 10, "035533": 10, "047110e": 10, "08": 10, "125248": 10, "778146": 10, "085258e": 10, "132698": 10, "4294": 10, "72": 10, "020243": 10, "611275e": 10, "84692": 10, "278146": 10, "032975e": 10, "93280": 10, "120": 10, "97": 10, "2506": 10, "059434": 10, "160646e": 10, "221442": 10, "800000": 10, "673793e": 10, "226454": 10, "current": 10, "sinc": 10, "underli": 10, "respons": 10, "anoth": 10, "sift": 10, "pool": 10, "contribut": 10, "pleas": 10, "selected_featur": 10, "len": 10, "column": 10, "36it": 10, "selct": 10, "kept": 10, "four": 10, "start": 10, "off": 10, "baselin": 10, "original_ma": 10, "581708285266024": 10, "test_featur": 10, "featurized_ma": 10, "f": 10, "round": 10, "243230501011035": 10, "absolut": 10, "error": 10, "instal": 10, "guid": 10, "releas": 10, "api": 10, "index": 10, "page": 10, "pypi": 11, "command": 11, "python": 11, "m": 11, "pip": 11, "clone": 11, "github": 11, "git": 11, "http": 11, "com": 11, "martineastwood": 11, "cd": 11}, "objects": {"featuristic.datasets": [[1, 0, 1, "", "fetch_cars_dataset"]], "featuristic.selection": [[2, 1, 1, "", "GeneticFeatureSelector"]], "featuristic.selection.GeneticFeatureSelector": [[2, 2, 1, "", "__init__"]], "featuristic.synthesis": [[3, 1, 1, "", "GeneticFeatureSynthesis"], [4, 1, 1, "", "SymbolicFunction"], [5, 0, 1, "", "list_operations"]], "featuristic.synthesis.GeneticFeatureSynthesis": [[3, 2, 1, "", "__init__"]], "featuristic.synthesis.SymbolicFunction": [[4, 2, 1, "", "__init__"]]}, "objtypes": {"0": "py:function", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "function", "Python function"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"api": 0, "refer": [0, 10], "genet": [0, 9, 10], "featur": [0, 9, 10], "gener": [0, 9], "select": [0, 2, 10], "dataset": [0, 1], "featurist": [1, 2, 3, 4, 5, 8, 10], "fetch_cars_dataset": 1, "geneticfeatureselector": 2, "synthesi": [3, 4, 5, 9, 10], "geneticfeaturesynthesi": 3, "symbolicfunct": 4, "list_oper": 5, "comput": 6, "perform": 6, "parsimoni": 6, "parallel": 6, "process": 6, "guid": 7, "us": 8, "With": 8, "scikit": 8, "learn": 8, "pipelin": 8, "load": [8, 10], "data": [8, 10], "split": 8, "train": [8, 10], "test": 8, "object": [8, 9], "function": [8, 9, 10], "fit": 8, "contain": 8, "access": 8, "insid": 8, "tune": 9, "complex": 9, "mathemat": 9, "express": 9, "max": 9, "earli": 9, "termin": 9, "popul": 9, "size": 9, "what": 10, "i": 10, "understand": 10, "quickstart": 10, "defin": 10, "cost": 10, "The": 10, "new": 10, "model": 10, "tabl": 10, "content": 10, "resourc": 10, "other": 10, "link": 10, "instal": 11, "sourc": 11, "releas": 12, "note": 12}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 60}, "alltitles": {"API Reference": [[0, "api-reference"]], "Genetic Feature Generation": [[0, "genetic-feature-generation"]], "Genetic Feature Selection": [[0, "genetic-feature-selection"]], "Datasets": [[0, "datasets"]], "featuristic.datasets.fetch_cars_dataset": [[1, "featuristic-datasets-fetch-cars-dataset"]], "featuristic.selection.GeneticFeatureSelector": [[2, "featuristic-selection-geneticfeatureselector"]], "featuristic.synthesis.GeneticFeatureSynthesis": [[3, "featuristic-synthesis-geneticfeaturesynthesis"]], "featuristic.synthesis.SymbolicFunction": [[4, "featuristic-synthesis-symbolicfunction"]], "featuristic.synthesis.list_operations": [[5, "featuristic-synthesis-list-operations"]], "Computational Performance": [[6, "Computational-Performance"]], "Parsimony": [[6, "Parsimony"]], "Parallel Processing": [[6, "Parallel-Processing"]], "Guides": [[7, "guides"]], "Using Featuristic With scikit-learn Pipelines": [[8, "Using-Featuristic-With-scikit-learn-Pipelines"]], "Load the Data": [[8, "Load-the-Data"], [10, "Load-the-Data"]], "Split the Data in Train and Test": [[8, "Split-the-Data-in-Train-and-Test"]], "Objective Function": [[8, "Objective-Function"], [9, "Objective-Function"]], "Fit a scikit-learn Pipeline Containing Featuristic": [[8, "Fit-a-scikit-learn-Pipeline-Containing-Featuristic"]], "Accessing Featuristic Inside the Pipeline": [[8, "Accessing-Featuristic-Inside-the-Pipeline"]], "Tuning the Genetic Feature Synthesis": [[9, "Tuning-the-Genetic-Feature-Synthesis"]], "Complexity of the Mathematical Expressions": [[9, "Complexity-of-the-Mathematical-Expressions"]], "Max Generations": [[9, "Max-Generations"]], "Early Termination": [[9, "Early-Termination"]], "Population Size": [[9, "Population-Size"]], "What is Featuristic?": [[10, "What-is-Featuristic?"]], "Understanding Genetic Feature Synthesis": [[10, "Understanding-Genetic-Feature-Synthesis"]], "Quickstart": [[10, "Quickstart"]], "Genetic Feature Synthesis": [[10, "Genetic-Feature-Synthesis"]], "Feature Selection": [[10, "Feature-Selection"]], "Define the Cost Function": [[10, "Define-the-Cost-Function"]], "The New Features": [[10, "The-New-Features"]], "Training a Model": [[10, "Training-a-Model"]], "Table of Contents": [[10, null]], "Resources & References": [[10, null]], "Other Links": [[10, "Other-Links"]], "Installation": [[11, "installation"]], "Source": [[11, "source"]], "Release Notes": [[12, "release-notes"]]}, "indexentries": {"fetch_cars_dataset() (in module featuristic.datasets)": [[1, "featuristic.datasets.fetch_cars_dataset"]], "geneticfeatureselector (class in featuristic.selection)": [[2, "featuristic.selection.GeneticFeatureSelector"]], "__init__() (featuristic.selection.geneticfeatureselector method)": [[2, "featuristic.selection.GeneticFeatureSelector.__init__"]], "geneticfeaturesynthesis (class in featuristic.synthesis)": [[3, "featuristic.synthesis.GeneticFeatureSynthesis"]], "__init__() (featuristic.synthesis.geneticfeaturesynthesis method)": [[3, "featuristic.synthesis.GeneticFeatureSynthesis.__init__"]], "symbolicfunction (class in featuristic.synthesis)": [[4, "featuristic.synthesis.SymbolicFunction"]], "__init__() (featuristic.synthesis.symbolicfunction method)": [[4, "featuristic.synthesis.SymbolicFunction.__init__"]], "list_operations() (in module featuristic.synthesis)": [[5, "featuristic.synthesis.list_operations"]]}})