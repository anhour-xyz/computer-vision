from collections import defaultdict


def findPokerHand(hand):
    ranks = []
    suits = []
    possibleRanks = []
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        if rank == "A": rank = 14
        elif rank == "K": rank = 13
        elif rank == "Q": rank = 12
        elif rank == "J": rank = 11
        else: rank = int(rank)
        ranks.append(rank)
        suits.append(suit)
        sortedRanks = sorted(ranks)

        
    # Royal Flush / Straight Flush / Flush
    if suits.count(suits[0]) == 5:
        if 14 in sortedRanks and 13 in sortedRanks and 12 in sortedRanks and 11 in sortedRanks:
            possibleRanks.append(10)
        elif all(sortedRanks[i] == sortedRanks[i-1] + 1 for i in range(1,len(sortedRanks))):
            possibleRanks.append(9)
        else:
            possibleRanks.append(6)

    # Straight
    check = True
    for i in range(1,len(sortedRanks)):
        if sortedRanks[i-1] + 1 != sortedRanks[i]:
            check = False
    if check == True: possibleRanks.append(5)

    # Four of a Kind / Full House / Three of a Kind / Two Pair / Pair
    map = defaultdict(int)
    for r in sortedRanks:
        map[r] += 1
    check = True
    if len(map) == 2:
        for value in map.values():
            if value != 4 and value != 1:
                check = False
        if check == True: possibleRanks.append(8)
        else: possibleRanks.append(7)
    elif len(map) == 3:
        check = True
        for value in map.values():
            if value == 3:
                check = False
        if check == True: possibleRanks.append(3)
        else: possibleRanks.append(4)
    elif len(map) == 4: possibleRanks.append(2)

    # High Card
    if not possibleRanks:
        possibleRanks.append(1)
    


    pokerHandRanks = {10:"Royal Flush",9:"Straight Flush",8:"Four of a Kind",7:"Full House", 6: "Flush", 5:"Straight",4: "Three of a Kind", 3:"Two Pair", 2:"Pair", 1:"High Card"}
    output = pokerHandRanks[max(possibleRanks)]
    print(hand, output)
    return output
    

if __name__ == "__main__":
    findPokerHand(["AH", "KH", "QH", "JH", "10H"]) # Royal Flush
    findPokerHand(["QC", "JC", "10C", "9C", "8C"]) # Straight Flush
    findPokerHand(["5C", "5S", "5H", "5D", "QH"]) # Four of a Kind
    findPokerHand(["2H", "2D", "2S", "10H", "10C"]) # Full House
    findPokerHand(["2D", "KD", "7D", "6D", "5D"]) # Flush
    findPokerHand(["JC", "10H", "9C", "8C", "7D"]) # Straight 
    findPokerHand(["10H", "10C", "10D", "2D", "5S"]) # Three of a Kind
    findPokerHand(["KD", "KH", "5C", "5S", "6D"]) # Two Pair
    findPokerHand(["2D", "2S", "9C", "KD", "10C"]) # Pair
    findPokerHand(["KD", "5H", "2D", "10C", "JH"]) # High Card

    

