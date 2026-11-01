#!/usr/bin/env python3
"""Render cumulative research figures from a frozen, derived JSON dataset.

Usage: python tools/render_progress_figures.py
Requires Matplotlib, NumPy and Pillow; no network, GPU or model calls.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/work-report-2026-11-01/report-data.json'
OUT=DATA.parent/'figures'
BLUE='#2563eb';ORANGE='#e87924';TEAL='#148a84';INK='#172b4d';GRAY='#607087';GRID='#e4eaf2'
WORKS=['short','long-context','long-decode'];WLABEL=['512 / 128','2048 / 256','512 / 512']
CONFIGS=['original3','one2048','one1024'];CLABEL=['MTP3 / chunk2048','MTP1 / chunk2048','MTP1 / chunk1024'];COLORS=[BLUE,TEAL,ORANGE]

def configure():
    for family in ['Hiragino Sans GB','Noto Sans CJK SC','Microsoft YaHei','DejaVu Sans']:
        try:font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False)
        except ValueError:continue
        plt.rcParams['font.family']=family;break
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':GRID,'text.color':INK,'xtick.color':GRAY,'ytick.color':GRAY,'axes.labelcolor':GRAY,'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def title(fig,heading,sub):
    fig.text(.07,.955,heading,fontsize=19,weight='bold',va='top')
    fig.text(.07,.885,sub,fontsize=10.5,color=GRAY,va='top')

def save(fig,name,note):
    fig.text(.07,.055,note,fontsize=9,color=GRAY,va='bottom')
    path=OUT/name;OUT.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=180);plt.close(fig)
    with Image.open(path) as im:
        clean=Image.new('RGB',im.size,'white');clean.paste(im.convert('RGB'));clean.save(path,optimize=True)
    print(path.relative_to(ROOT))

def baseline(d):
    fig,axes=plt.subplots(1,3,figsize=(13,5.3));fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.72,wspace=.27)
    for ax,w,label in zip(axes,WORKS,WLABEL):
        rows=sorted([r for r in d['baseline']['rows'] if r['workload']==w],key=lambda r:r['concurrency'])
        for mode,color,name in [('none',BLUE,'普通解码'),('mtp',ORANGE,'固定3步 MTP')]:
            x=[r['concurrency'] for r in rows];y=[r['modes'][mode]['output_throughput'] for r in rows]
            low=[min(r['modes'][mode]['round_output_tok_s']) for r in rows];high=[max(r['modes'][mode]['round_output_tok_s']) for r in rows]
            ax.plot(x,y,'o-',label=name,color=color,lw=2);ax.fill_between(x,low,high,color=color,alpha=.13)
            for xx,yy in zip(x,y):ax.annotate(f'{yy:.0f}',(xx,yy),xytext=(0,8 if mode=='mtp' else -14),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xscale('log',base=2);ax.set_xticks([1,8,32],['1','8','32']);ax.set_xlabel('客户端并发 C');ax.set_title(label+' token');ax.grid(axis='y',color=GRID);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('四卡整体输出（token/s）');fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.81),ncols=2,frameon=False)
    title(fig,'主基准：吞吐随并发的变化','SGLang 0.5.23 · TP=4 · 无 profiler · 三轮均值，阴影为三轮观测范围')
    save(fig,'01-baseline-throughput.png','来源：54 个正式基准窗口；仅连接 C=1/8/32 三个已测点，不表示连续并发模型或置信区间。')

def latency(d):
    fig,axes=plt.subplots(2,3,figsize=(13,7.2));fig.subplots_adjust(left=.075,right=.98,bottom=.17,top=.75,hspace=.42,wspace=.27)
    for col,(w,label) in enumerate(zip(WORKS,WLABEL)):
        rows=sorted([r for r in d['baseline']['rows'] if r['workload']==w],key=lambda r:r['concurrency'])
        for row,(metric,ylabel) in enumerate([('mean_ttft_ms','平均 TTFT（ms）'),('mean_tpot_ms','平均 TPOT（ms）')]):
            ax=axes[row,col]
            for mode,color,name in [('none',BLUE,'普通解码'),('mtp',ORANGE,'固定3步 MTP')]:
                ax.plot([1,8,32],[r['modes'][mode][metric] for r in rows],'o-',color=color,label=name)
            ax.set_xscale('log',base=2);ax.set_xticks([1,8,32],['1','8','32']);ax.grid(axis='y',color=GRID);ax.set_ylim(bottom=0)
            if col==0:ax.set_ylabel(ylabel)
            if row==0:ax.set_title(label+' token')
            else:ax.set_xlabel('客户端并发 C')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.835),ncols=2,frameon=False)
    title(fig,'主基准：首字等待与生成耗时','与图1使用相同的54个窗口；客户端计时，不代表 GPU kernel 时长')
    save(fig,'02-baseline-latency.png','来源：主基准逐模式三轮等权均值。TTFT 与 TPOT 分轴展示；各长度之间包含多个变化，不能作单因素归因。')

def tradeoff(d):
    fig,axes=plt.subplots(1,3,figsize=(13,5.6));fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.72,wspace=.26)
    for ax,w,label in zip(axes,WORKS,WLABEL):
        rows=sorted([r for r in d['baseline']['rows'] if r['workload']==w],key=lambda r:r['concurrency'])
        for mode,color,name in [('none',BLUE,'普通解码'),('mtp',ORANGE,'固定3步 MTP')]:
            x=[1000/r['modes'][mode]['mean_tpot_ms'] for r in rows];y=[r['modes'][mode]['output_throughput']/4 for r in rows]
            ax.plot(x,y,'o-',color=color,label=name)
            for r,xx,yy in zip(rows,x,y):ax.annotate('C'+str(r['concurrency']),(xx,yy),xytext=(4,5 if mode=='mtp' else -12),textcoords='offset points',fontsize=8)
        ax.set_title(label+' token');ax.set_xlabel('1000 / 平均 TPOT（token/s，换算）');ax.set_ylim(bottom=0);ax.grid(color=GRID)
    axes[0].set_ylabel('整体输出吞吐 / 4（token/s/GPU）')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.81),ncols=2,frameon=False)
    title(fig,'生成速度与系统吞吐的取舍','参考报告的双轴思路；本图横轴明确使用平均 TPOT 倒数，纵轴按四卡归一化')
    save(fig,'03-generation-throughput-tradeoff.png','横轴不是逐请求速率算术均值。计时含输入处理与排空，不能用 C×横轴直接还原纵轴；数据仍为同一主基准。')

def profile(d):
    rows=d['profile_rank0'];fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.subplots_adjust(left=.25,right=.97,bottom=.23,top=.74,wspace=.27)
    y=np.arange(len(rows));names=[r['configuration'].replace('long-context','2048/256') for r in rows]
    axes[0].barh(y,[r['target_ms'] for r in rows],color=[ORANGE if '-mtp' in r['configuration'] else BLUE for r in rows])
    for yy,r in zip(y,rows):axes[0].text(r['target_ms']+.25,yy,f"{r['target_ms']:.3f}",va='center',fontsize=9)
    axes[0].set_yticks(y,names);axes[0].set_xlim(0,40);axes[0].invert_yaxis();axes[0].set_title('普通 decode / target verify 活动')
    axes[1].barh(y-.13,[r['draft_ms'] or 0 for r in rows],height=.24,color=TEAL,label='draft')
    axes[1].barh(y+.13,[r['extend_ms'] or 0 for r in rows],height=.24,color=ORANGE,label='draft_extend')
    axes[1].set_yticks(y,['']*len(rows));axes[1].set_ylim(axes[0].get_ylim());axes[1].legend(frameon=False);axes[1].set_title('MTP 的额外活动，分别展示')
    for ax in axes:ax.set_xlabel('GPU 活动并集 / CPU 注释次数（ms/次）');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True)
    title(fig,'MTP 阶段活动：收益同时伴随执行成本','rank0 · 六个代表配置 · profiling 窗口；原始统计口径见正文')
    save(fig,'04-gpu-stage-activity.png','各柱不是可相加的请求关键路径；通信包含在阶段内。普通解码没有 draft 数据，右图对应位置为空。')

def ttft(d):
    fig,ax=plt.subplots(figsize=(11,5.8));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.73)
    arrivals=['单请求','同时32请求','固定4请求/秒'];configs=['普通解码','固定1步MTP','固定3步MTP'];x=np.arange(3)
    for shift,config,color in zip([-.24,0,.24],configs,[BLUE,TEAL,ORANGE]):
        values=[next(r['ttft_ms'] for r in d['ttft_control'] if r['arrival']==a and r['configuration']==config) for a in arrivals]
        bars=ax.bar(x+shift,values,.22,color=color,label=config)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set_xticks(x,['单请求\n每配置3个正式样本','同时32请求\n每配置3轮×32请求','固定4 req/s\n每配置3轮×64请求']);ax.set_ylabel('平均 TTFT（ms）');ax.set_ylim(0,1050);ax.grid(axis='y',color=GRID);ax.set_axisbelow(True)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),ncols=3,frameon=False)
    title(fig,'控制到达节奏后，固定1步与3步的首字延迟接近','同一512/128输入输出 · 27个控制窗口、873个正式请求 · 开启诊断计时')
    save(fig,'05-ttft-arrival-control.png','单请求仅3个样本/配置，属描述性诊断。不同到达条件的差值不能直接代替旧饱和闭环窗口中的排队成本。')

def goodput(d):
    cases=d['goodput']['cases'];fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.73,wspace=.25)
    for config,color,label in [('none',BLUE,'普通解码'),('fixed1',TEAL,'固定1步 MTP'),('fixed3',ORANGE,'固定3步 MTP')]:
        for ax,metric,scale in [(axes[0],'request_goodput',1),(axes[1],'attainment',100)]:
            values=[];lows=[];highs=[]
            for q in [4,7,9]:
                a=[c[metric]*scale for c in cases if c['label']==config and c['rate']==q]
                values.append(np.mean(a));lows.append(min(a));highs.append(max(a))
            ax.plot([4,7,9],values,'o-',color=color,label=label);ax.fill_between([4,7,9],lows,highs,color=color,alpha=.12)
    for ax in axes:ax.set_xlabel('计划周期到达率（req/s）');ax.set_xticks([4,7,9]);ax.set_ylim(bottom=0);ax.grid(color=GRID)
    axes[0].set_ylabel('有效吞吐（达标请求/s，含排空）');axes[1].set_ylabel('双门槛达标率（%，拒绝在分母）');axes[1].set_ylim(0,112)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.82),ncols=3,frameon=False)
    title(fig,'最大 token 吞吐之外：延迟达标能力','512/128 · 30秒计划输入 · 每点三轮 · TTFT≤500ms 且 TPOT≤30ms')
    save(fig,'06-goodput-arrival-rate.png','均值及阴影观测范围来自27个窗口；不是生产SLO保证。各到达率总请求数不同，门槛与统计口径不可省略。')

def sensitivity(d):
    rows=d['overnight_sensitivity'];fig,ax=plt.subplots(figsize=(10.8,5.6));fig.subplots_adjust(left=.085,right=.97,bottom=.22,top=.73)
    for config,color,label in [('fixed1',TEAL,'MTP1 / interval0'),('fixed3',BLUE,'MTP3 / interval0'),('interval8',ORANGE,'MTP1 / interval8')]:
        ax.plot([r['ttft_threshold_ms'] for r in rows],[r[config] for r in rows],'o-',color=color,label=label,lw=2)
    ax.axvline(500,color=GRAY,ls='--',lw=1);ax.set_ylim(-3,108);ax.set_ylabel('达标率（%）');ax.set_xlabel('复算 TTFT 门槛（ms），TPOT 仍固定30ms');ax.set_xticks([200,250,300,500]);ax.grid(color=GRID)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),ncols=3,frameon=False)
    title(fig,'短输入候选收益依赖允许的首字等待','已有夜间三轮复验 · 512/128、7 req/s、600秒周期到达；测后敏感性分析')
    save(fig,'07-slo-sensitivity.png','只复算同一批请求，不替换事前冻结的500ms/30ms主门槛；差异属于达标能力，不是同倍数的模型计算加速。')

def micro(d):
    rows=d['night']['micro_blocks'];fig,axes=plt.subplots(1,2,figsize=(12,5.5));fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.72,wspace=.23)
    for ax,n in zip(axes,['1','128']):
        x=np.arange(1,5);y=[r['groups'][n]['mean_ttft_ms'] for r in rows]
        for xx,yy,r in zip(x,y,rows):
            color=BLUE if r['configuration']=='one2048' else ORANGE
            ax.scatter(xx,yy,color=color,s=85);ax.text(xx,yy+4,f'{yy:.2f}',ha='center',fontsize=10)
        ax.set_xticks(x,['A1\n2048','B2\n1024','B3\n1024','A4\n2048']);ax.set_ylim(145,225);ax.set_ylabel('平均首响应（ms）');ax.set_title(f'每请求输出 {n} token');ax.grid(axis='y',color=GRID)
    title(fig,'独立执行第二次：单请求 ABBA 分块成本','相同16条2048-token输入 · 每配置2个独立服务block · MTP1、interval0、单在途')
    save(fig,'08-chunk-abba.png','每点16请求的均值；四点为服务block而非独立输入重复。1024-2048差值为+39.408ms（输出1）/+38.829ms（输出128）。')

def nightmatrix(d):
    works=['short-periodic','long-low-periodic','long-periodic','long-poisson'];labels=['512/128 · 周期4 req/s','2048/128 · 周期2 req/s','2048/128 · 周期4 req/s','2048/128 · 指数标称4 req/s']
    fig,axes=plt.subplots(1,3,figsize=(14,6));fig.subplots_adjust(left=.245,right=.975,bottom=.26,top=.72,wspace=.16)
    for ax,metric,scale,heading,cmap in zip(axes,['mean_ttft_ms','mean_tpot_ms','attainment'],[1,1,100],['平均 TTFT（ms）','平均 TPOT（ms）','双门槛达标率（%）'],['Blues','Oranges','YlGn']):
        array=np.array([[next(r['block_mean'][metric]*scale for r in d['night']['block_summaries'] if r['workload']==w and r['label']==c) for c in CONFIGS] for w in works])
        im=ax.imshow(array,aspect='auto',cmap=cmap,vmin=0,vmax=100 if metric=='attainment' else None)
        for y in range(4):
            for x in range(3):ax.text(x,y,f'{array[y,x]:.2f}',ha='center',va='center',color='white' if array[y,x]>.65*array.max() else INK,fontsize=10)
        ax.set_xticks(range(3),['3步\n2048','1步\n2048','1步\n1024']);ax.set_yticks(range(4),labels if ax is axes[0] else ['']*4);ax.set_title(heading)
    title(fig,'600秒负载复验：完整三轮结果','第二次独立执行 · 三个配置均为原生 interval0 · 三个服务block均值等权汇总')
    save(fig,'09-night-load-matrix.png','颜色仅在每个面板内比较；达标率包含拒绝，延迟仅覆盖成功请求。标称指数4 req/s的各轮实际计划率不同，不能跨行作单因素归因。')

def minutes(d):
    fig,axes=plt.subplots(1,3,figsize=(13,5.7));fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.72,wspace=.24)
    for ax,roundnum in zip(axes,[1,2,3]):
        for config,color,label in zip(CONFIGS,COLORS,CLABEL):
            r=sorted([c for c in d['minute_cohorts'] if c['round']==roundnum and c['label']==config],key=lambda c:c['arrival_minute'])
            ax.plot([c['arrival_minute'] for c in r],[c['mean_tpot_ms'] for c in r],'o-',ms=3.5,color=color,label=label)
        ax.axhline(30,color=GRAY,ls='--',lw=1);ax.set_ylim(20,68);ax.set_xticks([0,2,4,6,8]);ax.set_xlabel('计划到达分钟');ax.set_title('独立服务轮次 '+str(roundnum));ax.grid(axis='y',color=GRID)
    axes[0].set_ylabel('该到达 cohort 完整生命周期 TPOT（ms）')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.81),ncols=3,frameon=False)
    title(fig,'同一配置的第三轮变化必须保留','2048/128 · 周期4 req/s · 按计划到达分钟选取请求，跟踪完整生成生命周期')
    save(fig,'10-night-minute-cohorts.png','不是瞬时GPU速度或按完成分钟分桶。第三轮MTP1/2048出现变化，尚无匹配的系统时间线可确定根因。')

def main():
    configure();d=json.loads(DATA.read_text())
    for function in [baseline,latency,tradeoff,profile,ttft,goodput,sensitivity,micro,nightmatrix,minutes]:function(d)
if __name__=='__main__':main()
