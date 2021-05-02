def progress_bar(i, total):
    percent = round((i+1)*100/total, 2)
    j = round(percent/10)
    bar = '#'*(j) + "-"*(10-j)
    
    print(f'\r Progress: |{bar}| {percent}% Complete', end='\r')