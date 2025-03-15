num = int(input())

hour = num // 3600

minute = (num % 3600) / 60

second = (num % 3600) % 60

print(f"{int(hour)}시 {int(minute)}분 {int(second)}초")