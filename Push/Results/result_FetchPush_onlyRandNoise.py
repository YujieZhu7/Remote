import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'
x=np.arange(0, 2, 0.01)
#
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score2.npy")
BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RanNoise0.5/BC_S5_score.npy")
Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RanNoise0.5/Qfilter_S5_score.npy")
score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/First/EnsSize_10_S5_score.npy")

plt.plot(x, NoBC_score[:4000:20], color='black', label='NoBC')
plt.plot(x, BC_only_score[:4000:20], color='green', label='BC_only')
plt.plot(x, Qfilter_score[:4000:20], color='purple', label='Qfilter')
plt.plot(x, score[:4000:20], color='blue', label='Qfilter_EnsSize10')

plt.title('Scores')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Score')
plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_BC.png')
plt.show()

# demoAccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/First/EnsSize_10_S5_demoaccept.npy")
# demoAccept2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Mean/EnsSize_10_S5_demoaccept.npy")
# demoAccept3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Minimum/EnsSize_10_S5_demoaccept.npy")
# demoAccept4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/LCB/EnsSize_10_S5_demoaccept.npy")
# # demoAccept5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.5/LCB/EnsSize_20_S5_demoaccept2.npy")
# Qfilter_demo = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RanNoise0.5/Qfilter_S5_demoaccept.npy")
# plt.plot(x, demoAccept[11::200], color='blue', label='Qfilter')
# plt.plot(x, demoAccept2[11::200], color='red', label='Mean')
# plt.plot(x, demoAccept3[11::200], color='orange', label='Minimum')
# plt.plot(x, demoAccept4[11::200], color='green', label='LCB')
# plt.plot(x, Qfilter_demo[11::20], color='yellow', label='Qfilter_noensemble')
# # plt.plot(x, demoAccept5[1::20], color='purple', label='ModifiedLCB')
# plt.title('Acceptance rate of demonstrations')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Acceptance rate')
# plt.legend()
# # plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/demoaccept.png')
# plt.show()

success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/First/EnsSize_10_S5_success.npy")
success2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Mean/EnsSize_10_S5_success.npy")
success3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Minimum/EnsSize_10_S5_success.npy")
success4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/LCB/EnsSize_10_S5_success.npy")
# success5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.5/LCB/EnsSize_20_S5_success2.npy")
plt.plot(x, success[1::20], color='blue', label='Qfilter')
plt.plot(x, success2[1::20], color='red', label='Mean')
plt.plot(x, success3[1::20], color='orange', label='Minimum')
plt.plot(x, success4[1::20], color='green', label='LCB')
# plt.plot(x, success5[1::20], color='purple', label='ModifiedLCB')

plt.title('Success rate of demonstrations')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Success rate')
plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/success.png')
plt.show()

score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/First/EnsSize_10_S5_score.npy")
score2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Mean/EnsSize_10_S5_score.npy")
score3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/Minimum/EnsSize_10_S5_score.npy")
score4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RanNoise0.5/LCB/EnsSize_10_S5_score.npy")
# score5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.5/LCB/EnsSize_20_S5_score2.npy")
# BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S5_score.npy")
# Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S5_score.npy")
plt.plot(x, score[1::20], color='blue', label='Qfilter')
plt.plot(x, score2[1::20], color='red', label='Mean')
# plt.plot(x, score3[1::20], color='orange', label='Minimum')
plt.plot(x, score4[1::20], color='green', label='LCB')
# plt.plot(x, score5[1::20], color='purple', label='ModifiedLCB')
# plt.plot(BC_only_score[::20], color='black', label='BC_only')
# plt.plot(Qfilter_score[::20], color='purple', label='Qfilter_noensemble')
plt.title('score of demonstrations')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('score rate')
plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores.png')
plt.show()
