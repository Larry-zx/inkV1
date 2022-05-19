import time
from options.train_options import TrainOptions
from data.myDataLoader import myDataLoader
from model.cycle_model import CycleGANModel
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = myDataLoader(opt)
model = CycleGANModel(opt)
visualizer = Visualizer(opt)
total_steps = 0

# 迭代次数
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()  # 记录当前epoch开始时间
    epoch_iter = 0  # 记录iter次数

    # 遍历数据集
    for i, data in enumerate(data_loader):
        iter_start_time = time.time()  # 记录iter时间
        total_steps += opt.batchSize  # 总共的次数
        epoch_iter += opt.batchSize  # 当前epoch的次数
        model.set_input(data)
        if epoch >= opt.start_dec_sup:  # epoch到了sup权重要下降的时候
            model.lambda_sup = (1.0 * (200 - epoch) / (200 - opt.start_dec_sup)) * model.lambda_sup
        # 计算误差并更新参数
        model.optimize_parameters()
        # web可视化
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
        # 终端打印
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
