from .base_options import BaseOptions


class DemoOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./eval_results/', help='saves results here')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='demo')
        self.isTrain = False

        return parser
