import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import depth, get_y, iterate
from fine_dehalo import contrasharpening
from vardefunc.noise import Graigasm, AddGrain
from debandshit import placebo_deband, dumb3kdb

core = vs.core
src: vs.VideoNode = video_in

deband_args = dict(threshold=4, radius=16)

if src.height <= 810:
    deband_args.update(threshold=3, radius=12)


clip = depth(src, 16)

detail_mask = lvf.mask.detail_mask(clip, brz_a=0.045, brz_b=0.06)

pla_deband = placebo_deband(
    clip, grain=[4.0, 1.0], iterations=2, **deband_args
).std.MaskedMerge(clip, detail_mask)

contra_plac = contrasharpening(dumb3kdb(pla_deband), pla_deband, 2)

deband = dumb3kdb(contra_plac, threshold=[30, 24])

aa = lvf.aa.based_aa(deband, stg.misc.x56_SHADERS)

pref = iterate(get_y(deband), core.std.Maximum, 2).std.Convolution([1] * 9)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 90, 138)],
    strengths=[(1, 0.15), (0.65, 0.1), (0.35, 0.06)],
    sizes=(0.8, 0.75, 0.65),
    sharps=(55, 45, 40),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=True),
        AddGrain(seed=69420, constant=True)
    ]
).graining(deband, prefilter=pref)

grain = depth(deband, src.format.bits_per_sample)

grain.set_output()
