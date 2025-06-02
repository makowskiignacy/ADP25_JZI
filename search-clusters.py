from abc import ABC, abstractmethod

claster0 = {'c' : 1,
           'b' : 2,
           'a' : 3,
           'd' : 4}

claster10 = {'c' : 11,
           'b' : 12,
           'a' : 13,
           'd' : 14}


claster20 = {'c' : 21,
           'b' : 22,
           'a' : 23,
           'd' : 24}

claster30 = {'c' : 31,
           'b' : 32,
           'a' : 33,
           'd' : 34}

claster1 = {'g' : claster0,
            'f' : claster10,
            'e' : claster20}

claster2 = {'h' : claster1,
            'i' : claster1}


class CompoundSearch(ABC):
    @abstractmethod
    def similarity(self, compound1: str, compound2: str) -> float:
        pass

    @abstractmethod
    def cluster(self, compounds: list[str]) -> dict[str, dict[str]]:
        pass

# function to sort cluster alphabeticly by key
def similarity1(compound1, compound2):
    return ord(compound1)


#creates a lists of keys and clasters for sorting by key
def dict_to_list(dict):
    keys = dict.keys()
    dict_list = []
    for key in keys:
        dict_list.append({'key' : key, 'claster': dict[key]})
    return dict_list
    


#gets n most similar compunds to given compunds with given similarity function
def get_similar_compounds(compound, claster, similarity_func, n):
    if False == isinstance(claster,dict):
        return [claster]
    claster_list = dict_to_list(claster)
    def sorting_func(a):
        return similarity_func(a['key'], compound)
    claster_list.sort(key = sorting_func)
    n_remain = n
    compound_list = []
    for i in claster_list:
        new_compounds = get_similar_compounds(compound, i['claster'], similarity_func, n_remain)
        n_remain -= len(new_compounds)
        compound_list += new_compounds
        if n_remain<=0:
            return compound_list

    return compound_list

print(get_similar_compounds('a', claster2, similarity1, 10))
#chyba działa dobrze


