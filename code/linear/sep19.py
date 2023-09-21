def maxProfit(prices):
    leftIdx, rightIdx = 0,1
    profit = 0
    # lowest = 0
    while(rightIdx < len(prices)):
        if prices[leftIdx] >= prices[rightIdx]:
            leftIdx += 1
            rightIdx += 1
        else:
            gain = prices[rightIdx] - prices[leftIdx]
            profit = max(gain, profit)
            rightIdx += 1
    return profit


# print(maxProfit([7,1,5,3,6,4]))
print("Hi")