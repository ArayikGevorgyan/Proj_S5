def print_summary(stats_dict):
    print("\n--- Patient Data Summary ---")
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            print(f"{key}: Mean={value['mean']}, Min={value['min']}, Max={value['max']}")
        else:
            print(f"{key}: {value}")
