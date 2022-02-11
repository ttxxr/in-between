test_conf = {
    "data": {
        "data_path": "C:/Users/alantxren/ubisoft-laforge-animation-dataset/lafan1/lafan1_bvh",
        "save_bvh": False,
        "save_gif": True,
        "save_metric": True,
        "test_nums": "single",
        # "single_idx": [9],
        "single_idx": [9, 98, 99, 102, 123],
        "flip": True,

        "sequence_length": 50,
        "joint_num": 22,
        "state_input_dim": 95,
        "offset_input_dim": 91,
        "target_input_dim": 88,
        "short_length": 2,
        "long_length": 10,
        "joints_to_remove": [
            5,
            10,
            16,
            21,
            26
        ],
        "offsets": [
            [-42.198200, 91.614723, -40.067841],
            [0.103456, 1.857829, 10.548506],
            [43.499992, -0.000038, -0.000002],
            [42.372192, 0.000015, -0.000007],
            [17.299999, -0.000002, 0.000003],
            [0.000000, 0.000000, 0.000000],

            [0.103457, 1.857829, -10.548503],
            [43.500042, -0.000027, 0.000008],
            [42.372257, -0.000008, 0.000014],
            [17.299992, -0.000005, 0.000004],
            [0.000000, 0.000000, 0.000000],

            [6.901968, -2.603733, -0.000001],
            [12.588099, 0.000002, 0.000000],
            [12.343206, 0.000000, -0.000001],
            [25.832886, -0.000004, 0.000003],
            [11.766620, 0.000005, -0.000001],
            [0.000000, 0.000000, 0.000000],

            [19.745899, -1.480370, 6.000108],
            [11.284125, -0.000009, -0.000018],
            [33.000050, 0.000004, 0.000032],
            [25.200008, 0.000015, 0.000008],
            [0.000000, 0.000000, 0.000000],

            [19.746099, -1.480375, -6.000073],
            [11.284138, -0.000015, -0.000012],
            [33.000092, 0.000017, 0.000013],
            [25.199780, 0.000135, 0.000422],
            [0.000000, 0.000000, 0.000000]
        ],
        "parents": [-1, 0, 1, 2, 3, 4,
                    0, 6, 7, 8, 9,
                    0, 11, 12, 13, 14, 15,
                    13, 17, 18, 19, 20,
                    13, 22, 23, 24, 25],
        "foot_index": [9, 10, 11, 12, 13, 14,
                       21, 22, 23, 24, 25, 26],
    },
    "model": {
        "model_name": "cvae_bi2",
        "position_encoding_basis": 10000,
        "noise_theta": 0.5,
        "latent_size": 16,
        "encoder": {
            "state_dim": [
                95,
                512,
                256
            ],
            "offset_dim": [
                91,
                512,
                256
            ],
            "target_dim": [
                88,
                512,
                256
            ]
        },
        "lstm": {
            "lstm_dim": [
                768,
                1536
            ],
            "layer_num": 1
        },
        "decoder": {
            "decoder_dim": [
                1536,
                512,
                256
            ],
            "out_dim": 91,
            "contact_dim": 4
        }
    },
    "network": {
        "batch_size": 32,
        "epoch": 2000,
        "learning_rate": 0.001,
        "beta1": 0.5,
        "beta2": 0.9,
        "save_path": "",

        # first: cvae bi2
        "load_path": "C:/Users/alantxren/Desktop/DockerFiles/12-15-cvae_bi2/checkpoint/"
                     "cvae_bi2@2021-12-15-14_57_17",
        "epoch_num": "320",

        "quaternion_loss_weight": 1,
        "position_loss_weight": 0.5,
        "contact_loss_weight": 0.1,
        "root_loss_weight": 1,
        "gan_loss_weight": 0.1,
        "kld_weight": 1.0,
        "velocity_weight": 0.25,
        "slide_loss_weight": 0.1,

        "save_duration": 20,
        "weight_decay": 0.00001,
    }
}
