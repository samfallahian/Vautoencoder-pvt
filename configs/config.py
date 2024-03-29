CFG = {
    "data": {
        "path": "/mnt/d/sources/data/DL-PTV/merged/",
        "speeds": [0.06528, 0.06852, 0.07824, 0.09768, 0.10092, 0.11064, 0.12036, 0.13008, 0.16248, 0.17868],
        "folders": ["3p6", "4p4", "4p6", "5p2", "6p4", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"],
        "folder": "3p6_distance_full",
        "start_frame": 200,
        "end_frame": 260,
        "split_fraction": 2,
        "test_size": 0.3,
        "feature_no": 6
    },
    "train": {
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "buffer_size": 1000,
        "epochs": 60,
        "H": 256,
        "H2": 128,
        "latent_dim":8,
        "print_iteration": 2,
        "dropout": 0.3,
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}

# CFG_example = {
#     "data": {
#         "path": "/mnt/d/sources/data/DL-PTV/merged/",
#         "speeds": [0.06528, 0.06852, 0.07824, 0.09768, 0.10092, 0.11064, 0.12036, 0.13008, 0.16248, 0.17868],
#         "folders": ["3p6", "4p4", "4p6", "5p2", "6p4", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"]
#     },
#     "train": {
#         "batch_size": 64,
#         "buffer_size": 1000,
#         "epoches": 20,
#         "val_subsplits": 5,
#         "optimizer": {
#             "type": "adam"
#         },
#         "metrics": ["accuracy"]
#     },
#     "model": {
#         "input": [128, 128, 3],
#         "up_stack": {
#             "layer_1": 512,
#             "layer_2": 256,
#             "layer_3": 128,
#             "layer_4": 64,
#             "kernels": 3
#         },
#         "output": 3
#     }
# }
