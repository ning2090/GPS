import numpy as np
import matplotlib.pyplot as plt
import sys, os, csv
import pandas as pd
from gnssutils import EphemerisManager
from datetime import datetime, timezone, timedelta
import folium
import matplotlib
matplotlib.use('TkAgg')

# 第一步： 处理gnss观测数据
ff = 'logdata/gnss_log_2023_12_02_23_35_46.txt'
try:
    f = open(ff, 'r')
    l = []
    for raw in f.readlines():
        if '#   Raw' in raw or '# Raw' in raw:
            title = raw.replace('\n', '').replace('# ', '')
        if '#' not in raw and 'Raw' in raw:
            raw2 = raw.replace('\n', '')
            l.append(raw2)
    f.close()
    name_list = title.split(',')[1:]
    dic = {}
    for name in name_list:
        dic[name] = []
    for i in l:
        ll = i.split(',')[1:]
        for j in range(len(ll)):
            dic[name_list[j]].append(ll[j])
    pd.DataFrame(dic).to_excel('raw_data.xlsx', index=None)
    print('done')
except:
    f.close()
    f = open(ff, 'r')
    l = []
    for raw in f.readlines():
        if '#   Raw' in raw or '# Raw' in raw:
            title = raw.replace('\n', '').replace('# ', '')
        if '#' not in raw and 'Raw' in raw:
            raw2 = raw.replace('\n', '')
            l.append(raw2)
    f.close()
    name_list = title.split(',')[1:-1]
    dic = {}
    for name in name_list:
        dic[name] = []
    for i in l:
        ll = i.split(',')[1:-1]
        for j in range(len(ll)):
            dic[name_list[j]].append(ll[j])
    pd.DataFrame(dic).to_excel('raw_data.xlsx', index=None)
    print('done')

# # 第二步： 计算观测伪距
raw_data = pd.read_excel('./raw_data.xlsx')
c = 2.99792458e8  # 光速 单位m/ns
WEEK_SEC = 604800  # 周化秒
# 计算GNSS接收时间，考虑偏差和偏移
t_rx_gnss = raw_data['TimeNanos'] + raw_data['TimeOffsetNanos'] - raw_data['FullBiasNanos'].iloc[1] - \
            raw_data['BiasNanos'].iloc[1]
weekNumber = np.floor(t_rx_gnss * 1e-9 / WEEK_SEC)
t_Rx = 1e-9 * t_rx_gnss - WEEK_SEC * weekNumber
t_Tx = 1e-9 * raw_data['ReceivedSvTimeNanos'] + raw_data['TimeOffsetNanos']
time_diff = t_Rx - t_Tx
Pseudorange_GPS = c * time_diff #计算伪距
id_ = raw_data['Svid']
type_ = raw_data['ConstellationType']
dic = {'type': type_, 'Svid': id_, 'GPS': Pseudorange_GPS}
df = pd.DataFrame(dic)
df = df[(df['GPS'] > 1e6) & (df['GPS'] < 1e8)] #筛选伪距
df.to_excel('伪距.xlsx', index=None)
print('done')


# 第三步：筛选GPS卫星的信息
weiju = pd.read_excel('伪距.xlsx')
weiju = weiju[weiju['type'] == 1] #GPS type=1
weiju.to_excel('伪距_after.xlsx', index=None)
print('done')

#第四步：格式化原始数据raw_data
raw_data = pd.read_excel('./raw_data.xlsx')

# 格式化卫星ID
raw_data['Svid'] = raw_data['Svid'].apply(lambda x: f'0{x}' if len(str(x)) == 1 else x)

# 格式化卫星ConstellationType
raw_data.loc[raw_data['ConstellationType'] == 1, 'Constellation'] = 'G'
raw_data.loc[raw_data['ConstellationType'] == 3, 'Constellation'] = 'R'
# 只保留Constellation为G的数据
raw_data = raw_data.loc[raw_data['Constellation'] == 'G']
raw_data['SvName'] = raw_data['Constellation'] + raw_data['Svid'].astype(str) #定义SvName格式并创建SvName列
# 将某些列转换为数字表现形式
raw_data['Cn0DbHz'] = pd.to_numeric(raw_data['Cn0DbHz'])
raw_data['TimeNanos'] = pd.to_numeric(raw_data['TimeNanos'])
raw_data['FullBiasNanos'] = pd.to_numeric(raw_data['FullBiasNanos'])
raw_data['ReceivedSvTimeNanos'] = pd.to_numeric(raw_data['ReceivedSvTimeNanos'])
raw_data['PseudorangeRateMetersPerSecond'] = pd.to_numeric(raw_data['PseudorangeRateMetersPerSecond'])
raw_data['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(raw_data['ReceivedSvTimeUncertaintyNanos'])
# 检查BiasNanos和TimeOffsetNanos列是否存在
if 'BiasNanos' in raw_data.columns:
    raw_data['BiasNanos'] = pd.to_numeric(raw_data['BiasNanos'])
else:
    raw_data['BiasNanos'] = 0
if 'TimeOffsetNanos' in raw_data.columns:
    raw_data['TimeOffsetNanos'] = pd.to_numeric(raw_data['TimeOffsetNanos'])
else:
    raw_data['TimeOffsetNanos'] = 0
# 计算GpsTimeNanos并将其转换为Unix时间格式
raw_data['GpsTimeNanos'] = raw_data['TimeNanos'] - (raw_data['FullBiasNanos'] - raw_data['BiasNanos'])
gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
raw_data['UnixTime'] = pd.to_datetime(raw_data['GpsTimeNanos'], utc=True, origin=gpsepoch)
raw_data['UnixTime'] = raw_data['UnixTime']
raw_data['t_Tx'] = t_Tx
raw_data['weekNumber'] = weekNumber
raw_data['Pseudorange_GPS'] = Pseudorange_GPS

# 根据时间间隔分割数据
raw_data['Epoch'] = 0
raw_data.loc[raw_data['UnixTime'] - raw_data['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
raw_data['Epoch'] = raw_data['Epoch'].cumsum()
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)

# 第五步：获取星历
# 使用EphemerisManager模块从NASA官网寻找对应时间的星历文件
manager = EphemerisManager(ephemeris_data_directory)
epoch = 0 #历元初始化
num_sats = 0 #卫星数量初始化
# 找到至少 5 颗卫星
while num_sats < 5:
    one_epoch = raw_data.loc[(raw_data['Epoch'] == epoch) & (time_diff < 0.1)].drop_duplicates(subset='SvName')
    timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
    one_epoch.set_index('SvName', inplace=True)
    num_sats = len(one_epoch.index)
    epoch += 1

sats = one_epoch.index.unique().tolist()
ephemeris = manager.get_ephemeris(timestamp, sats) # 检索指定时间戳和卫星列表的星历数据
print(timestamp)
print(one_epoch[['UnixTime', 't_Tx', 'weekNumber']])

# 第六步：根据星历数据定位卫星位置
def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    # sv_position来存储卫星位置信息
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)
    # 计算从参考历元开始的经过时间
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    # 计算平均异常并迭代求解偏心异常
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1
    # 定义三角函数
    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)

    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    # 卫星时钟的时间校正
    delT_oc = transmit_time - ephemeris['t_oc']
    # sv_position['delT_sv']: 卫星时钟校正
    # ephemeris['SVclockBias']：传输时的卫星时钟偏差
    # ephemeris['SVclockDrift']：卫星时钟漂移的速率
    # delT_oct_oc：卫星时钟时间()与传输时间之间的时间差
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))
    # 纬度自变量
    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)
    # 对轨道元素校正
    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k
    # 校正的轨道元素
    u_k = Phi_k + du_k
    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k
    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']
    # 计算轨道平面中的坐标
    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)
    # 校正升交点经度
    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']
    # 计算地心地球固定 (ECEF) 坐标
    sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['z_k'] = y_k_prime * np.sin(i_k)
    return sv_position

sv_position = calculate_satellite_position(ephemeris, one_epoch['t_Tx'])
# 使用计算出的卫星时钟校正来调整伪距信息
pr = one_epoch['Pseudorange_GPS'] + 2.99792458e8 * sv_position['delT_sv']
dic = {'Svid': list(one_epoch.index), 'X': list(sv_position['x_k']), 'Y': list(sv_position['y_k']),
       'Z': list(sv_position['z_k']), 'GPS': list(pr)}
df = pd.DataFrame(dic)
df.to_excel('merged_data.xlsx', index=None)


# 第七步： 最小二乘法+牛顿迭代
data = pd.read_excel('merged_data.xlsx')
# 删除数据为空的行
data.dropna(axis=0, how='any', inplace=True)
# x_,y_,z_表示卫星的位置
x_list = list(data['X'])
y_list = list(data['Y'])
z_list = list(data['Z'])
# D表示GPS卫星的观测伪距，# 这里用平方，省去了开根号的复杂运算
D_list = list(data['GPS'] ** 2)
c = 299792458e-9

# 构造卫星位置的矩阵
def get_G(x, y, z, t, x_list, y_list, z_list):
    G_list = []
    for i in range(len(x_list)):
        raw = []
        # 关于x的偏导数
        raw.append(2 * (x - x_list[i]) + 2 * c * t * (1 / 2) * (
                (2 * (x - x_list[i])) / ((x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2) ** (
                1 / 2)))
        # 关于y的偏导数
        raw.append(2 * (y - y_list[i]) + 2 * c * t * (1 / 2) * (
                (2 * (y - y_list[i])) / ((x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2) ** (
                1 / 2)))
        # 关于z的偏导数
        raw.append(2 * (z - z_list[i]) + 2 * c * t * (1 / 2) * (
                (2 * (z - z_list[i])) / ((x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2) ** (
                1 / 2)))
        raw.append(
            2 * c * (((x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2) ** (1 / 2)) + c ** 2 * 2 * t)
        G_list.append(raw)
    return np.array(G_list)

# 根据两点间的距离公式计算接收器到卫星之间对的距离f，存储在f_list中
def get_f(x, y, z, t, x_list, y_list, z_list):
    f_list = []
    for i in range(len(x_list)):
        f = (x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2 + 2 * c * t * (
                ((x - x_list[i]) ** 2 + (y - y_list[i]) ** 2 + (z - z_list[i]) ** 2) ** (1 / 2)) + (c * t) ** 2
        f_list.append(f)
    return np.array(f_list)

# 伪距D_list-接收器和卫星之间的距离f_list = 误差B_list
def get_B(D_list, f_list):
    B_list = []
    for i in range(len(D_list)):
        B_list.append(D_list[i] - f_list[i])
    return np.array(B_list)

# 状态初始化
d_list = [1]  # 增量
xx_list = [0]
yy_list = [0]
zz_list = [0]
tt_list = [0]
ii = 0
# 输入一个循环，一直持续到最后一个元素d_list小于1e-3或达到最大迭代次数10000
while d_list[-1] > 1e-3 and ii < 10000:
    x = xx_list[-1]
    y = yy_list[-1]
    z = zz_list[-1]
    t = tt_list[-1]
    f_list = get_f(x, y, z, t, x_list, y_list, z_list)
    B = get_B(D_list, f_list)
    G = get_G(x, y, z, t, x_list, y_list, z_list)
    # 使用超定非线性方程最小二乘法最小化残差
    d = np.linalg.inv(G.T.dot(G)).dot(G.T).dot(B) # d：增量调整的向量
    # 计算向量的欧几里德范数，判断当前迭代步的收敛情况
    dt = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + d[3] ** 2) ** (1 / 2)
    d_list.append(dt)
    xx_list.append(x + d[0])
    yy_list.append(y + d[1])
    zz_list.append(z + d[2])
    tt_list.append(t + d[3])
    print(ii)
    print(d_list[-1])

    ii = ii + 1

plt.figure()
plt.plot(range(1, len(xx_list) + 1), xx_list)  # , marker='o'
plt.plot(range(1, len(yy_list) + 1), yy_list)  # , marker='o'
plt.plot(range(1, len(zz_list) + 1), zz_list)  # , marker='o'
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel("Position")
plt.title('Position .VS. Iteration')
plt.legend(['x', 'y', 'z'])
plt.savefig('Position.png')

plt.figure()
plt.plot(range(1, len(d_list) + 1), d_list)  # , marker='o'
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel("Increment")
plt.title('Increment .VS. Iteration')
plt.savefig('Increment.png')

from pyproj import Proj, transform
# 接收器坐标转换，从XYZ坐标系到经纬度坐标系的转换
def xyz_to_latlon(x, y, z):
    # 定义源坐标系（XYZ坐标）
    src_proj = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    # 定义目标坐标系（经纬度坐标）
    dest_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 进行坐标转换
    lon, lat, _ = transform(src_proj, dest_proj, x, y, z)
    return lat, lon, _

# 示例坐标
x = xx_list[-1]
y = yy_list[-1]
z = zz_list[-1]

print(x, y, z)

# 转换坐标
latitude, longitude, _ = xyz_to_latlon(x, y, z)
# 打印结果
print(f"经度：{longitude}, 纬度：{latitude}, 高度：{_}")

# 创建地图对象，中心定位在中国
china_map = folium.Map(location=[35, 105], zoom_start=4)
# 在地图上标记坐标点
folium.Marker(
    location=[latitude, longitude],
    popup='Your Location',
    icon=folium.Icon(color='red')
).add_to(china_map)
# 保存地图为HTML文件
china_map.save('china_map.html')
