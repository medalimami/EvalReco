from .VirtualAlgo import VirtualAlgo
import openai
import time
from typing import List, Tuple


class AlgoGPT(VirtualAlgo):
    def __init__(self, api_key):
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo-1106"
        self.prompt = ""
        self.last_call_time = time.time()-1

    def fit(self, trainset: List[Tuple[int, int, float]]):
        self.prompt = "\nGiven this user-item rating matrix, predict the rating for the following User and Movie and make sure to return a rating that's ONLY either 1.0 , 2.0 , 3.0 , 4.0 or 5.0!" \
                      "Don't explain your approach or how you managed to get the answer just return the rating in your message! nothing else. your answer has to be ONLY either 1.0 , 2.0 , 3.0 , 4.0 or 5.0!\n"

        rating_matrix = "\n".join([f"User{user} gives Movie{item} a rating of {round(rating)}" for user, item, rating in trainset])

        self.prompt = rating_matrix + self.prompt

    def estimate(self, u: int, i: int) -> float:
        
        # Si moins de 0.1 secondes se sont écoulées depuis le dernier appel, on attend
        time.sleep(max(0, 1 - (time.time() - self.last_call_time)))
        self.last_call_time = time.time()
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt + f"\nFor User {u} , Movie {i} what would you predict as a rating? "}
            ],
            temperature=0
        )

        predicted_rating = response['choices'][0]['message']['content']

        #print(f"Predicted rating for User {u} for Movie {i}: {predicted_rating}")

        return float(predicted_rating)

    def get_base_model_name(self) -> str:
        return self.__class__.__name__