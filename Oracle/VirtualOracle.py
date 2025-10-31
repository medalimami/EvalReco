
from typing import List, Tuple

# This is a virtual oracle class that will be used to generate the data for the oracle
class VirtualOracle:
    # This method will be used to generate the data for the oracle
    def get_rating(self, id_user: int, id_item: int) -> float:
        assert False, "get_rating not implemented"
        pass
    
    def get_users(self) -> List[int]:
        assert False, "get_users not implemented"
        pass
    
    def get_items(self) -> List[int]:
        assert False, "get_items not implemented"
        pass

    # Return a random [user, item, rating]
    def get_random_exemple(self) -> List[Tuple[int, int, float]]:
        assert False, "get_random_exemple not implemented"
        pass
    
    def get_number_of_ratings(self) -> int:
        assert False, "get_number_of_ratings not implemented"
        pass