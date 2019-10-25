import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--module', type=str, required=True)

    parser.add_argument('--content_dir', type=str, required=False,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, required=False,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=500)

    # testing options
    parser.add_argument('--decoder', type=str, default='experiments/decoder.pth.tar')
    parser.add_argument('--content', type=str, 
                         help='File path to the content image')
    parser.add_argument('--style', type=str,
                        help='File path to the style images')
    parser.add_argument('--content_size', type=int, default=512,
                        help='New size for the content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New size for the style image')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output images')
    # advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The weight that controls the degree of stylization. Should be between 0 and 1')
    parser.add_argument('--style_interpolation_weights', type=str, default='',
                        help='The weight for blending the style of multiple style images')
    
    return parser.parse_args()