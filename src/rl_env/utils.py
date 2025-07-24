# You can add helper functions for reward calculation or observation building here
def calculate_reward(buy_price, sell_price):
    return sell_price - buy_price
