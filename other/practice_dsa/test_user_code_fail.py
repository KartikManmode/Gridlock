
def FindIndex(n, nums, target):
    l=0
    r=n-1
    res=n
    
    # Loop 1
    l=0
    r=n-1
    while(l<=r):
        mid=(l+r)//2
        if nums[mid]==target:
            if mid<res:
                res=mid 
            l=mid+1
        elif nums[mid]<nums[mid-1]:
            l=mid+1
        else:
            if nums[mid]<target:
                l=mid+1
            elif nums[mid]>target:
                r=mid-1
    
    # Loop 2
    l=0
    r=n-1
    while(l<=r):
        mid=(l+r)//2
        if nums[mid]==target:
            if mid<res:
                res=mid 
            r=mid-1
        elif nums[mid]>nums[mid-1]:
            r=mid-1
        else:
            if nums[mid]>target:
                l=mid+1
            elif nums[mid]<target:
                r=mid-1
                
    return res

# Test case where it fails: Strictly decreasing array
nums3 = [5, 4, 3, 2, 1]
target3 = 4
n3 = len(nums3)
result3 = FindIndex(n3, nums3, target3)
print(f"Test 3 - Array: {nums3}, Target: {target3}")
print(f"Expected: 1, Got: {result3}")
