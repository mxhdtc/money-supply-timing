# coding = utf-8
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import os
import matplotlib.pyplot as plt
import calendar
from math import *

# 参数设置
init_asset = 1  # 初识资产的净值
start_time = '2011-11-01'  # 计算央行货币净投放量起始时间，日期为当月1号，输入格式为YYYY-MM-DD
start_time_1 = '2012-12-01'  # 择时策略的起始时间,日期为当月1号,输入格式为YYYY-MM-DD
end_time = '2020-05-31'  # 结束时间,日期为当月最后一天,输入格式为YYYY-MM-DD
path = '/Users/maxiaohang/指数数据/'  # 资产数据存放的文件夹路径
path_macro = '/Users/maxiaohang/宏观数据.xlsx'  # 宏观数据的存放路径
path_rate = '/Users/maxiaohang/利率数据.xlsx'  # 利率数据的存放路径
path_moneysupply = '/Users/maxiaohang/货币投放数据.xlsx'  # 货币投放数据的路径
path_openmatket = '/Users/maxiaohang/公开市场操作数据.xlsx'  # 公开市场操作数据的路径
days = 250  # 每年交易日天数
non_risk_rate = 0.015  # 无风险利率
asset_num = 12  # 资产个数
period = 12  # 计算均值的窗口期


class Timing():
    def __init__(self, asset_data, asset_name, signal_payoff, period=period, start_time=start_time_1,
                 end_time=end_time, init_asset=init_asset):
        # asset_data 资产的净值数据
        # asset_name 基金的名称
        # signal_payoff 用于构造信号的资产的,需要提前计算好
        # risk为无风险利率
        # days为每年交易日天数
        # period为计算均值的窗口期
        self.init_asset = init_asset
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.asset_data = asset_data
        self.signal_payoff = signal_payoff
        self.asset_name = asset_name
        self.period = period
        self.rate = non_risk_rate
        self.days = days
        self.asset_payoff = self.get_asset_payoff(self.asset_data, asset_name=asset_name)

    def get_asset_payoff(self, asset_data, asset_name):
        # 获取资产的日收益率
        start_time_1 = str(pd.to_datetime(self.start_time) - relativedelta(months=1))[0:7]
        end_time_1 = str(pd.to_datetime(self.end_time) + relativedelta(months=1))[0:7]
        asset_data = asset_data[
            (asset_data[asset_data.columns[0]] <= end_time_1) &
            (asset_data[asset_data.columns[0]] >= start_time_1) &
            (asset_data[asset_data.columns[1]] != '--')]
        asset_data = asset_data.rename(index=asset_data[asset_data.columns[0]],
                                       columns={asset_data.columns[1]: asset_name})
        asset_data[asset_data.columns[1]] = asset_data[asset_data.columns[1]] \
            .apply(lambda x: str(x).replace(',', '')).copy(deep=True)
        asset_data[asset_data.columns[1]] = pd.to_numeric(asset_data[asset_data.columns[1]], errors='coerce')
        asset_data.index = pd.to_datetime(asset_data.index)
        return asset_data[asset_data.columns[1]].pct_change().iloc[1:]

    def moving_average_strategy_signal(self, month, monthly_payoff):
        # 按照1年期国债均线择时策略判断当前月份是否选择持有资产
        # monthly_payoff是月收益率，其索引为每月的最后一天
        # month为当月最后一天
        month = pd.to_datetime(month)
        mean_payoff = monthly_payoff[(monthly_payoff.index > month - relativedelta(months=self.period)) &
                                     (monthly_payoff.index < month + relativedelta(months=1))].mean()
        month_payoff = monthly_payoff[month]
        return month_payoff < mean_payoff

    def moving_average_strategy(self, init_asset, payoff, signal_payoff):
        # 计算1年期国债均线择时策略下资产每日的净值
        # payoff是资产的日收益率，索引是每个交易日
        # 返回策略每个交易日的净值
        cumulative_income = init_asset
        net_value = []
        flag_summary = [] # 计算每期信号值
        payoff = payoff[(payoff.index >= self.start_time) &
                        (payoff.index <= self.end_time)]
        index = payoff.resample('M').apply(lambda x: x[-1]).index  # 取每月最后一天
        for month in index:
            if self.moving_average_strategy_signal(month, signal_payoff):
                flag = 1
            else:
                flag = 0
            for day_payoff in payoff[(payoff.index >= str(month)[0:7]) &
                                     (payoff.index <= month)]:
                cumulative_income = cumulative_income * np.power(1 + day_payoff, flag) * \
                                    np.power(1 + self.rate/self.days, 1 - flag)
                # 当月持有资产择按日资产收益率计算净值；不持有资产则按照当月1年期国债收益率计算当日净值
                net_value.append(cumulative_income)
                flag_summary.append(flag)
        return pd.DataFrame({'择时'+self.asset_name+'净值': net_value, 'flag': flag_summary}, index=payoff.index)

    def none_strategy(self, init_asset, payoff):  # 没有择时策略的净值曲线
        # payoff是资产的日收益率，索引是每个交易日
        cumulative_income = init_asset
        net_value = []
        payoff = payoff[(payoff.index >= self.start_time) &
                        (payoff.index <= self.end_time)]
        index = payoff.resample('M').apply(lambda x: x[-1]).index
        for month in index:
            for day_payoff in payoff[(payoff.index >= str(month)[0:7]) &
                                     (payoff.index <= month)]:
                cumulative_income = cumulative_income * (1 + day_payoff)
                net_value.append(cumulative_income)
        return pd.DataFrame({self.asset_name+'净值': net_value}, index=payoff.index)

    def Max_Drawdown_ration(self):
        # 返回一个长度为2的tuple，第一个元素为择时策略的最大回测，第二个元素为资产的最大回测
        n1 = self.moving_average_strategy(self.init_asset, self.asset_payoff,
                                               self.signal_payoff)  # n1是择时策略的每日净值
        n2 = self.none_strategy(self.init_asset, self.asset_payoff)  # n2是不择时下资产的每日净值
        n1 = np.array(list(n1[n1.columns[0]]))
        n2 = np.array(list(n2[n2.columns[0]]))
        n1_drawdowns = []
        n2_drawdowns = []
        for i in range(len(n1)):
            n1_min = np.min(n1[i:])
            n2_min = np.min(n2[i:])
            n1_drawdowns.append(1 - n1_min / n1[i])
            n2_drawdowns.append(1 - n2_min / n2[i])
        return np.max(n1_drawdowns), np.max(n2_drawdowns)


    def plot_net_value(self):
        # 分别画出择时策略和标的资产的净值曲线
        timing_payoff = self.moving_average_strategy(self.init_asset, self.asset_payoff, self.signal_payoff)
        non_timing_payoff = self.none_strategy(self.init_asset, self.asset_payoff)
        plt.plot(timing_payoff.index, timing_payoff[timing_payoff.columns[0]], 'b',
                 non_timing_payoff.index, non_timing_payoff[non_timing_payoff.columns[0]], 'r')
        plt.legend(labels=['timing ' + self.asset_name, self.asset_name], loc='best')
        plt.show()

    def Sharpe_ratio(self):
        # 返回一个长度为2的tuple，第一个元素为择时策略的Sharpe ratio，第二个元素为资产的Sharpe ratio
        n1 = self.moving_average_strategy(self.init_asset, self.asset_payoff,
                                               self.signal_payoff)  # n1是择时策略的每日净值
        n1 = n1[n1.columns[0]]
        n2 = self.none_strategy(self.init_asset, self.asset_payoff)  # n2是不择时下资产的每日净值
        exReturn1 = np.array(n1.pct_change().iloc[1:])
        exReturn2 = np.array(n2.pct_change().iloc[1:])
        return (np.mean(exReturn1)-self.rate / self.days)/np.std(exReturn1)*np.sqrt(self.days),\
               (np.mean(exReturn2)-self.rate / self.days)/np.std(exReturn2)*np.sqrt(self.days)


class MoneySupply_Timing(Timing):
    # signal_payoff 为央行货币净投放量
    def __init__(self, asset_data, asset_name, signal_payoff, start_time=start_time_1,
                 end_time=end_time, init_asset=init_asset):
        # asset_data 资产的净值数据
        # asset_name 基金的名称
        # signal_payoff 用于构造信号的资产的,需要提前计算好
        super(MoneySupply_Timing, self).__init__(asset_data, asset_name, signal_payoff, start_time=start_time,
                                                 end_time=end_time, init_asset=init_asset)

    def moving_average_strategy_signal(self, month, monthly_value):
        # monthly_value是用来判断择时信号的月频数据
        # month为每月的最后一天
        month = pd.to_datetime(month)
        start_month = month - relativedelta(months=self.period)
        index = monthly_value.index
        mean = np.mean(monthly_value[(index >= start_month) & (index < month)]).values[0]
        std = np.std(monthly_value[(index >= start_month) & (index < month)]).values[0]
        return (monthly_value[index == month].iloc[0, 0] - mean) / std

    def moving_average_strategy(self, init_asset, payoff, signal_payoff):
        # 计算央行货币净投放量择时策略下资产每日的净值
        # payoff是资产的日收益率，索引是每个交易日
        # 返回策略每个交易日资产的净值
        cumulative_income = init_asset
        net_value = []
        payoff = payoff[(payoff.index >= self.start_time - relativedelta(months=1)) &
                        (payoff.index <= self.end_time)]
        index = payoff.resample('M').apply(lambda x: x[-1]).index  # 取每月最后一天
        inidcator = [self.moving_average_strategy_signal(i, signal_payoff) for i in index]
        flag = 0.5 * np.sign(inidcator[0]) + 0.5  # 初始信号值大于0持仓1，小于0空仓
        for i in range(1, len(index)):
            if inidcator[i] > 1 or inidcator[i] < -1:
                flag = 0.5 * np.sign(inidcator[i]) + 0.5
            elif inidcator[i - 1] * inidcator[i] < 0:  # 本月信号值介于[-1,1]且与上月信号值异号，仓位调至0.5
                flag = 0.5
            month = index[i]
            for day_payoff in payoff[(payoff.index >= str(month)[0:7]) &
                                     (payoff.index <= month)]:
                cumulative_income = cumulative_income * (1 + flag * day_payoff +
                                                         self.rate * (1 / self.days) * (1 - flag))
                # 当月持有资产择按日资产收益率计算净值；不持有资产则按照无风险利率计算当日净值
                net_value.append(cumulative_income)
        return pd.DataFrame({payoff.name: net_value}, index=payoff[(payoff.index >= self.start_time) &
                                                                   (payoff.index <= self.end_time)].index)


def get_file_name(path, filetype):  # 读取文件的名字
    file_name = []
    os.chdir(path)
    for root, dir, files in os.walk(os.getcwd()):
        for file in files:
            if os.path.splitext(file)[1] == filetype:
                # print(os.path.splitext(file)[1])
                file_name.append(file)
    return file_name


if __name__ == '__main__':
    filetype = '.xls'
    file_name = get_file_name(path, filetype)
    file_name = np.array(file_name)
    file_name.sort()  # 存放资产的名称
    macro_data = pd.read_excel(path_macro).iloc[2:]  # 导入宏观数据
    asset_data = [pd.read_excel(path + fname) for fname in file_name]  # 导入资产数据，放到一个列表里，每一个元素是一个dataframe
    moneysupply_data = pd.read_excel(path_moneysupply).iloc[2:].fillna(0)  # 导入货币投放量数据
    openmarket_data = pd.read_excel(path_openmatket).iloc[2:].fillna(0)  # 导入公开市场操作数据
    openmarket_data = openmarket_data.rename(index=openmarket_data[openmarket_data.columns[0]])
    openmarket_data.index = pd.to_datetime(openmarket_data.index)
    openmarket_data['month'] = openmarket_data.index
    openmarket_data['month'] = openmarket_data['month'].apply(
        lambda x: pd.to_datetime(str(x.year) + '-' + str(x.month) + '-' + str(x.days_in_month)))
    openmarket_month = openmarket_data.groupby('month')[openmarket_data.columns[1]].sum()  # 先计算每月的公开市场操作加总
    openmarket_rolling_month = [openmarket_month.iloc[i - 3:i].sum()
                                for i in range(3, openmarket_month.shape[0])]
    openmarket_rolling_month = pd.DataFrame(openmarket_rolling_month,
                                            columns=['滚动三个月公开市场操作'],
                                            index=openmarket_month.index[3:])  # 获取滚动三个月的公开市场操作数据

    moneysupply_data = moneysupply_data.rename(index=moneysupply_data[moneysupply_data.columns[0]])
    moneysupply_data = moneysupply_data[list(moneysupply_data.columns[1:5])]
    moneysupply_rolling_month = [(moneysupply_data.iloc[i - 1] - moneysupply_data.iloc[i - 4]).sum()
                                 for i in range(4, moneysupply_data.shape[0])]
    moneysupply_rolling_month = pd.DataFrame(moneysupply_rolling_month,
                                             columns=['滚动三个月货币投放'], index=moneysupply_data.index[4:])
    moneysupply_rolling_month = moneysupply_rolling_month[moneysupply_rolling_month.index >= start_time]
    openmarket_rolling_month = openmarket_rolling_month[openmarket_rolling_month.index >= start_time]
    net_moneysupply = [moneysupply_rolling_month.loc[i, '滚动三个月货币投放']
                       + openmarket_rolling_month.loc[i, '滚动三个月公开市场操作'] for i in moneysupply_rolling_month.index]
    net_moneysupply = pd.DataFrame(net_moneysupply,
                                   columns=['央行货币净投放滚动三个月'], index=moneysupply_rolling_month.index)
    rate_data = pd.read_excel(path_rate)  # 导入利率数据
    rate_data = rate_data.iloc[2:]
    rate_data = rate_data.rename(index=rate_data[rate_data.columns[0]])
    rate_1y_data = rate_data[rate_data.columns[1]].dropna()
    rate_10y_data = rate_data[rate_data.columns[2]].dropna()
    rate_1y_data = rate_1y_data.resample('M').apply(lambda x: np.mean(x)).iloc[1:]
    rate_10y_data = rate_10y_data.resample('M').apply(lambda x: np.mean(x)).iloc[1:]
    asset_timing = [
        MoneySupply_Timing(asset_data[i], 'asset_{} net value'.format(i + 1), net_moneysupply) for i in
        range(asset_num)]
    i = 0
    win_ratio = []
    Sharpe_timing = []  # 择时夏普率
    Sharpe_non = []  # 不择时夏普率
    Drawdown_timing = []  # 择时最大回撤
    Drawdown_non = []  # 不择时最大回撤
    for asset in asset_timing:
        d1 = asset.none_strategy(asset.init_asset, asset.asset_payoff)
        d2 = asset.moving_average_strategy(asset.init_asset, asset.asset_payoff, asset.signal_payoff)
        df = pd.concat([d1, d2], axis = 1)
        s = (df[df.columns[0]]<df[df.columns[1]])
        win_ratio.append(len(s.loc[s==True])/len(s))
        asset.plot_net_value()#画出全部12个资产的净值曲线
        df.to_csv('/Users/maxiaohang/Desktop/liangxin_results/{}(央行货币净投放量择时).csv'.format(file_name[i].split('.')[0]))  # 导出各资产择时净值与flag
        i = i+1
        Sharpe_timing.append(asset.Sharpe_ratio()[0])
        Sharpe_non.append(asset.Sharpe_ratio()[1])
        Drawdown_timing.append(asset.Max_Drawdown_ration()[0])
        Drawdown_non.append(asset.Max_Drawdown_ration()[1])
    win_ratio = pd.DataFrame({'央行货币净投放量择时胜率': win_ratio}, index = [file_name[i].split('.')[0] for i in range(len(file_name))])
    win_ratio.to_csv('/Users/maxiaohang/Desktop/liangxin_results/{}.csv'.format('央行货币净投放量择时胜率'))  # 导出各资产择时胜率
    summary = pd.DataFrame(np.array([Sharpe_timing, Sharpe_non, Drawdown_timing, Drawdown_non]).T,
                           columns=['择时夏普率', '不择时夏普率', '择时最大回撤', '不择时最大回撤'],
                           index=[file_name[i].split('.')[0] for i in range(len(file_name))])
    summary.to_csv('/Users/maxiaohang/Desktop/liangxin_results/{}.csv'.format('央行货币净投放量择时表现'))



    # 测算代码可忽略
    # for asset in asset_timing:  # 画出全部12个资产的净值曲线
    #     asset.plot_rate_moving_average_strategy()
    # self = asset_timing[0]
    # self.none_strategy(self.init_asset, self.asset_payoff)
    # self.moving_average_strategy(self.init_asset, self.asset_payoff, self.signal_payoff)
