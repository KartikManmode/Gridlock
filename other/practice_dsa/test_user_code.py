
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
        elif mid > 0 and nums[mid]<nums[mid-1]:
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
        elif mid > 0 and nums[mid]>nums[mid-1]:
            r=mid-1
        else:
            if nums[mid]>target:
                l=mid+1
            elif nums[mid]<target:
                r=mid-1
                
    return res

# Testing with a classic valley array
# [5, 4, 3, 2, 3, 4, 5], target = 4
# Expected leftmost index: 1
nums = [5, 4, 3, 2, 3, 4, 5]
target = 4
n = len(nums)
result = FindIndex(n, nums, target)
print(f"Test 1 - Array: {nums}, Target: {target}")
print(f"Expected: 1, Got: {result}")

# Testing with target at index 0
# [10, 8, 6, 7, 9], target = 10
nums2 = [10, 8, 6, 7, 9]
target2 = 10
n2 = len(nums2)
result2 = FindIndex(n2, nums2, target2)
print(f"Test 2 - Array: {nums2}, Target: {target2}")
print(f"Expected: 0, Got: {result2}")
