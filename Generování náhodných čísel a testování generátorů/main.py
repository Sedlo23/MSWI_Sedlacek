import random

# Získání vstupu od uživatele
user_input = input("Zadejte libovolný text:")

# Převod vstupu na číselnou hodnotu pomocí funkce hash()
seed = hash(user_input)

# Nastavení semínka pro generátor pseudonáhodných čísel
random.seed(seed)

# Generování náhodného čísla
random_number = random.random()

print("Vygenerované náhodné číslo:", random_number)