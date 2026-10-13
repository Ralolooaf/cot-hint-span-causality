# =============================================================================
# Study 3 -- CELL 1 of 2: tier-1 results (Kaggle run of 2026-10-08, code c6f0f3f31e9f3b81)
# =============================================================================
# Writes the session-1 run directory (1149 records) into rjy_runs/study3/ so that
# CELL 2 (study3_mediation.py) continues with tier 2. Every file is rebuilt and
# checked against the SHA-256 of the file Kaggle produced; on any mismatch nothing
# is written. If results already exist there (a rerun), this cell does nothing.
import base64, hashlib, json, lzma
from pathlib import Path

RUN_DIR = Path("rjy_runs/study3")
SHA256 = {
    "olmo-3-7b-think.jsonl": "d36f8ddc1aa3a08b2d27671763c8e27bc672335b8f3591904dd5c516d5ce2ec9",
    "olmo-3-7b-think.surgery.jsonl": "c555a05667053786e5cb1931941c3433c6ce6a59dc47f794a8f4d7f482852200",
    "olmo-3-7b-think.selection.jsonl": "3eaa2cf8183e87908a62feda90445371d79d78ce5a9ddfa6af6990b79a8eed0c",
    "prereg_study3.json": "4ffc8b518b5610409f4e785959f13ef55c7485427e664efc51b0be36a656226a",
    "report_study3.md": "835cf674dda54853a8ed69a1e5925069b513178d68ac08528699903c980db0d5"
}
BLOB = """
{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;sQ5Q^j!cwh>Av&$p?6H4}?>Y=YXD}AX^UXf7`~(E7%#`LXZQX$v~F?+~>SpbiU0ZYjSiU
Dt$@rpaD_1LEUVbedqXTeSi~(`cIu0V<4NbGXSAVr(<=O<w6R?p|#PT{X8m4RFjSmyj=*DHX&+iLh+*A#GArDZ>dmuGPd>}hIvjX
bKXn_9HFE!0E1ci=}o?~cTU957bO`71A4%mU&u~m%^e{{R&gdBetv;@bw?lX?!+{9qG{p63Qw<0_q#`r>g8&8Oq1Ka0z|((@o|Wm
zIHNSmz26e#Ov0o-zhD&@+z4X^=CfN5E=##Cxg>^`~X<9QcCSo{{{&V{+8V=IlB;m<XZ2y6XLt=$z~%S@9)K{atz*v4di#FoDI&B
xV3iw!;nvHut1-<rEzbJ&3Vz?4oxI|gh5!+mxwb(g0e11D3*>DQC(kGoR?QWI!oyOVzF_geb~FhBLhg8h_YF}zFIT9cDdetkqQy}
9CR2F12G0^enCR2H{a@2FRI?dvH^SJ)l-hfb_t*fg6lH|f}E$WGF<^2ug&!Wvyi|b{+Pv^zezXzB|_M(6jc6I@YS>ca|l0t7xK?f
?<WeOz8JXp?Y%7HM4DiutT<u^AyqCqD3~JG18o<zZ;r@%&x0sE-=)@Vxr|rP(QhC?hg7JxziPk*%wlJzOjE&PJnW0L{}8%VY3dsh
C8YnyJ;**iR`yqxFkYb-m5!8UfQySXp&{^xZWBzLGD;<2RpN+qr;=uvQub8kC{$kC{r`L2lRxLolw$&|gAlXZZtL+hag_CxA|Y5c
X(_-~ATXIbiHdo~Um|jEG`%_V1X31rL77E2Ar9DHFrdc2Nnt2^n$LDQN{OsLF$vTW3Jts8iz%m!%>AV3*js1=+}X82)cJjb744!^
Ub-1kTEdG-E@4XWTb8m;S(0@(Bq;v65mJ&~P9BHAjuzFAy;UHDxz^}#3DmziM$mFKU%ZE07o%SOeT0t~lHxbrLah9==<huh4t^mU
Z+h}Cq4>(D;vOIYx3DSD`Ic^TY9S(!faw4z6i6PAwq@tQ82sx@r5SSb{LJ^v6zOkz?<E&0K)Yz}po#QRG;#{M0$7RLaK4SSOU<}(
K8I~{E=h2vvW$6PWO;qFlL|7Wx@yXL;yWXxOguHcTgDDDn#zEmrK8x_+>PR1;$mY7zE>z-an}<co$yeP_9=hp&PUZf4&PyCu9X&r
8haHJPuAt|MZCrF@$D7l!-dXjk!X^i%zE^qJ4-<TwcOI4Uz`V?*wt(gaspmqRevnxOCUUigUst_$=bjZGeN?W>g4UUaIy^_lVdmt
qBvhk7qysk!70t^7(%H|CS?=Oe&`Qb^42_ITs8sfGHUHIi!C<8=OCIVEroEhP$^+)$SRE94Q!rk{x>i8EeFdM_>3%yBR!Q;t3uV%
!>mc57>d*&s6jINHzCLBuI_GhU=L_ZX-@o3`H%U#CX8XIjh+8cq|fu8s3K?J^1sES4~VS-zMaxmWqt(ldGviuLyE4bT8uhLxL^Aq
QI**EL+-v(-_9%l`*1-|n+tNxbaNz}o_?e@t8^+~qB~tpf%@#oRM4&7>A2+hX;*vGq&yh7cd6NI#rsD8iQs*aR6FRXCc^Ut!Y_MX
Vl)~&t!OrH%ed8A&B+DAn+uUg<1*BM&4s-Ax5DG2_7@?d8uB_E&>8-z87f_ml}vp!8eKCC5q#@A70a2CN8LeJn45v{s_bvU^M8x{
^pYNEXyev3Xb*7;LLax5U$Hi3Z3i5Li)dh{0?O3Q0wH0N6qgS_cwt71H`MA*kU{45Myz94TM}tRX^9cvl%TvfxaTyPq_kcEa#Gb8
D>%Nk9}u8b3*r?I2`k`a8F;}hid=P11puucnK{EL$k6jnlYRdY-@uRU`fcA$bc$|GZ_Mx}LTeVvBTzV4TSzZHcBl<(0NJN==yQkW
7;*2UiEsE=DgX3)pcz;n$5)@KA~42F@aldl(!3`ffq@hd40xpK1<wBk_y7LaNDXr>O^`mpSQL`T{y{zHXf35U*<yduvqWsYbJw+q
Bwr3#F-_>Roe+n{ingO?<ZN#F?zja^Ie3V3{$XQo*c~7r0g;47AvUsjRN`{iV<xXf;{b7SG$b!&1#2{6=BgX>rx&+v6qu<2DF#!N
8P?omyl$U}K39v{HPMg|sOeTO<-A+r;Vn7TxxI5qKWho2>z2hEsDtz9IT*^MZYgNCOd9Pt4&(L0%ekHG3L{>xf8(~aIyKabYzijs
kB7)FiLZa5Az=KgcmX*x189|w4BPh*p+FoUQXM2%8&g@<vRT=y_2@1JcXUaMn@reVanuTX@x|39Whk_TTci!Y{cpMT_;Bd79>xtd
Vh`tV(y0R`Z+PWYilP=Cpi^j&4t%{4wC(JlZY9Vz@i>nZYW&pv*ZHf_x7H-ouBpYpFWYlcp%p;Y#~%4od7y0|k1sCY!Li)cO<~|3
CFk!@NY$L~u&MN!n^I1REf_xHOQE|Kj`cKDjMW@C2;LI<DgEZXX^6Y3>JhT5A!Syuc+8!lykPpWdH*vi7dy|ChO-n}h4tT~$Bl-y
x)|17q5SZ{qv)W$a?cj#lI`c{;jWTueRmVpqDZ#lA6`Ip?6r|-%P?429N8Wu%O2#*(Is-r_yI?5rZH6t+b~noq52a&X%pLPrj2h&
hBy0FByB~a8g=kF(y-blR-uYfB*Gy*#xlJZ%GU^QdK{U^IvVTJ|3jw6#j%BizdTpLC`g9Qda}g`{z&ltkg?m;`C5J$epTM-%SZLE
azRBoW+xx9v<=OVQ;4qV1a5S>UAGE=#HW=wo~P^ii?Mhp%On-@McSd>P3l7(%@y{^vp_BkUmd%eILbBaw^gQtbD0!VJHmD@DPp}e
-721;L2>h_DpC2Iu&1tt9(EQjicyr|1KH<$ms5mhoNn_%m;O2DR5D@>u!gF{<*C9=eKNi-BDSN<`VSboe;J+oe0Mnog2{iTb_)xE
EPx^U4^#tvw5mOdTGs0b%l*E>4D%=ydtK|$73fk1OU(uOJ*$&EMIa~rp(+(IKCl_bT7>4fK2W=$XUy)P%@AbP*YF~;uSl9&vs6d=
xIutQ>ZTbx!!I|Gq=m<hw})&9zx#Q36#9~c0sO1#Cv(Qeja45xqyLg~1?5kpf>h+es_B#p0o?`$^gCEbPS-Y%(}}YkaIu(>0|alM
l-&wI&Sfz~>|O|fDd@c0@I3&6pb1g!^zvY(^%mnZozCa~lpskeeXBz+g8R+zsjn4&K5Lz6)jcX6mI9q#(CwR<^%#VQ_On*0iR(+?
c!Y*E<jFWGsmcMLaO#|C)#p)%05UKhVP^(bjP&O3Ih-l3SVzK0fl0RDPT&NK9~Q6LGe}+$(F}Wq=^xCRx-jsKM`-uvbwf1l88&OM
Nd8OMvUtW72vxWrNKXqr%P+PW-qnTi#;2oDX=uhzcP2+{Ohkgl+0%ekYD<O@oMEJf^70YLlZfJm0c+ELsKoJgnZTrjW~cM?)Bq;7
qNetg3vl2tE@M1mzMfTq`Rw}y+?*MF=LL8R8>-L}{q0}nAYYl(^lBL!OB<?I12-z6;XYASF6?sBt}U@1k{e0H!pLB_-oy8B7z10h
eFayjLs@77@sTLeT+U9Qk61PYy(X+qCw8T7C~1CVJ}X-l<Cv=4#A_^IlKGRiMyej3&D@-bz76Kh?cuN)kk@uQ#BFZfOLaqwuEg`X
yX^y>rY+V2s59bpYAF7!D7g4Pw-F;_jNp1$vv!GCsy6cSOu0o-4l&j1kD_hh68>cO$p+oy;VRiTw;(wR&u5J((GQ1zi;o|hN#5$t
>N+f>jeK-$&Brm+z}7om(pU@e`K7n5Q+|+wZXs^dw8o%DGiJfP6l||SrGR!+PoHsWPlMY=Q2Pk1zx?a{A=FOqmW~&qID0xHV3%Pa
ONF5V9-KvRJG;<hKe%6>)R@IuUDS3>VnY@Ev{!h}bP3_vc2UM4vs|6Jy4Fq5ip+Z&13Q~>sJPZV)Gc>py#HAV_`%3ftHEe-)G22s
->NL<Nawf$enMCr`UsN(1zWd9OeV7O2~eY|fpx-DmU<T!ICnENcQRF2Wcskg^VDPvmfrj;Pvtr=#u`jF?Dyrimu;Fz9}N?Fn>P{x
=L4c-WW}$>cA~z3)>x1VkWRqA2gTXqU&$^|+jhkF?1oZHwgP?d+wiH_s3XYiuev-avOfjMF;rzq|691=!DiI9<VuMJE?zJYjWs7$
TVQagFE_s%oZIk3b3Ozi#44cko(_PDocHC1Z5;rRb_BLHTMXWtmbVt7*W(6He_4j_m<{P{38+c!SOC^3^*vYqUwmZa+Qo42M-1g%
s&GDfL`R;%Yi<Ev^=C!H&#AoCAfP{_GNlRcMYydJBcI5|JO>0-2;*Q2V6~9|#*o(%qVTs?qXxcjLG+22A*58Oq$~2eFnlhX!f~d|
DQM6*)Kt_Hz3`SkCVh^VYnY-e1`$uSk+&f~{^}8S`7-j}u65Lw#<>Aajf~0_bzq3cMc^9wvq>N5*_)3>DCl0<*E1AGc~ho*a(`Al
BAPu+J4&Gq^^hyc2V-frx>ps;mP+hyU<fq8U4$>!qzl{Ik;)%KI_4J;F?@K9))ap;NO|2l6TM|NjIj-$7h$!AlKqr6N;=99`-KC{
oU+jbMB^<M;oP7*R_enO=A}gq@kJkWBqwQ(dDh=4b*(`NVH8ZC(t~=NTOQ^7U%pPj3pZ2DtbB6FSIJKSbr?uph7zt+kvuM+@Vt+P
A2a3gWAx@^J#mbW+Q|>U<=n%D#R`S3&#X1JH!PbBbz65NI)3c1P&UvcspE$ZcQcC_iR-#xQxb*W%e+ms+(r`=663M{b+oWG3PoQ&
&184IgGo8GPofl09{7%*V&L#M<-M1MmxMJ&LRiJA%`&9k`TZM_Iq_0{<e)eBUAW~;!ZnXe6!FcBBMSz4eoo@I65!9kQh;?C)>gY&
!i=}hKx`84zoI<U=<If`n!0nvCEvp9v|xgiINNt=!g-S}qU*e8)P^)z{`AzDu5?$E1gy*gC<%k5d0=E6N`8RIcLvr1i%;6^gRli3
U~&QH=l(9(z2g;h7fIM#pKCmLPi5NA+n8h=;flgcT0K*W+vh2+EqpIFyBj4W#2FNEPcfc=em7_oc_NID;Lk9uG61ngVJf=<F#VwB
hcv+wKcUJ>qg77dD#F!^(64SxGyqrg4XFG_p^WW6?H&G<7}QFLor(`@8ZgZ?peN1hO&5DlFl|Ig8<j5VjaJd|*cdQovcT3z?UgBn
i+<YPHAb}eA8hS;q=pg#B1c3%FlSm|{{!2OU6;a@@}VIKs%Q(`o?>%^_!yNje9JT4d2*QR3eEV+hD_porP5bx84~|VwdkMlm16VO
8_6J_M$TC6D7XKv5+^0mj%CQ$Pr4gy0sVt6KBv(;cv|Kq^acWvYA^>Zqq3q*YOlf;3@>doQ2>egBLKaj=B~3esMS1fgp&s9LGlLY
^%uJ{yN~Ua2L>f}X$=SP(<G2jQ!8pDVVy{jYDm(@FMg|&l?RS{EBsxeN>9dZn#p>*Nj{N(CB;{>-2WIQwNbwe4u;HBu9Q&$$CV8n
CM7{LzA77m-|%QmB-kx0@be$p%XMkNgDVnP%ye_u(8xe+a-a}v5i;J+lL0JUsreKT4^^FbQtP87>E&Nn_hYe{-w&C1Ygi`|yW@|y
@naNCHt0;SkHg6@nRd#qCW@z1gCjFej#sBkZu!nruj+?=aD-Dio)~}u3c`td?TQa#D;XjJTLpqE8M8VE986z<w1dc^c?}ft?~CHE
OO;2<6Eum@gSdA}ys6VSz+$P;K3ep7DZ}bQ5mk;vEsR}J8!FPN$zZ8fPEt)eFJ>1mP;q15Q}{df1a`67$~uX77-#Kx=Dj4ZZq1B`
$Q(*|)%$_N!H^dx-YvKngr|_(mPhlc3_CrEjUO%HF;vexJn-9P|Ea<Vq$opJAvbe1DLMD4Hiz0hq;G9EtOm-Qlu82jn?(U25zqA)
h+%V!Mh0|pZ<+ihZ3~s8Xm+FrjCC2^#NW9>DcS+k&UyAD?BjMI24CWye>5~T;;p#i101g#7&iDRo;3|eH6f(+#=9YnQkV;K{EvV8
${f2IytI>|6|(&S5I$31f-~dvPs^|3c~FIoK_won8EibcUBgX=9ml}c=10(>6s(b2n_lf>^uLLv+2j*KpLR!P@KtkLEw5F;W-I_R
%bBsuT9C3vA}^(s8vfGqPE?<%5vw^{^*BhzU6qU5#^Fh$bo>l^0f&gYI*D<$h+j7TU`aR_=BZxNXUhwYsNXdoIomN^BBWb_t=ItR
s<d0+7(lM?0fTsT!7h-OX1V{MCV3dvJBxJFYx%%QvT<4=`S7zLd}N_9tgDbj{MXU6AVj(A(brQ%qske7jOV{OJca4Z@PkS!fz<^W
E7els%E1P%GA6G1h}Q`Jiac_^D8SZNh&)>FEo%^P&`fi1x5#})nh2~}!-S{IFGX%GBhK=`>{)rBPQu?W^6dYMOpj<mN4v&Y^TO;V
V}}{A&dJzwT2pfVacp$tMvESFi<+ruJ9r;wl(|YEIJn)^8C50n5xFTo`5~v5)c-UFxKK7<&v!+eb4?Q((;E+R_mFi#ZR!TD7Z4ok
vd~9*TZ1(J`0N*~IEjnaMxOHoy3A(G*$mDK(<5@!<3R6k@S?jE2?42LqX2Vcv;VARRSis|j1}flf&2}u?xJ!{<$c~wulE9y7XPj)
Laj1nKpHqq{sTq|Kusko(Th-`_3{B__U&giu`Qw>I?kn9?Dayv*r*QE135eWoazI<!GXR$KjeWHRyhk~X=Tf%Z*YMQh`YuJ!bL2h
WjDQisA0@kQ9UeA^ABrMGGA9TZ;AT7p(7Eo<KKZzTtqIZ9%Kr<f9`*Shs}F2Cr%?E#D1NWz08XOIZjH6aQzvFV4qC7hQEV@;RCm}
gy!-8j~NzTz`1hmK1H)3N&{bq5GL45-N52z4yQiX<!c)svVrnrTui1J{J6Bzzc1-|6x&Ygx)Nkh!t%#ZZ#O9o#Z>zEVLW41*rtJi
8nD>BpR9D)z31{sXsjEi-WjxL0@+0=WT|>ejN}3V^aYcz0QTn6_`zG&3u{1KCdM64=z!MEF14x))m6ox;lVI83jR`8l=_^B)<>HM
0X;G|rw03~PKv=(`YU{<;W{FK^YIMJv%5JXXzXg)G`4s9{R~4}{)^)ef)p}SkS4OI7vV)}HLFG82$C=lHu3#kmOS*rn^~#*g~Qzp
Tf3>fb!5}>?Z|MAHWVSAoste~iBA09=Hd1^%GNcfvUMgd@42Agfm~99dS3tB{+Tmi&Swpu%7Z_?k-lQi1I7Fk5GP`vwoOj6$qb0<
+WEjr#v&LQm*fJh#d_KFY3xW2(8A)kE+hFD$ypd8goM@b?+Wvd4w-am`-Cp=zC+lxWg+8Xp(xL8%IQPswN5KcwB$<l@Q`lJ?y-hG
c&Vs*nObC$bk*d5fg*qyTe_i-8gv1P-OY-n(f!3GCmm_AbFf}U>Jyzd;L<Pg7c7$cIaZ-W^0LI9DhG({<$ip4e{H6?Sc4|(0pA=@
^3a<CbRplLne5Ntcf=l1m_pyfe!T6(M=O^Vd%tUAqGMSy6)lq**{)IB0Uj;0e5YHyERxvGh-axB(0qXXQo$=PJ}lmpD@?Z9#1bHk
P~VHPjzvPk^v{m`uI)**JC!}n85CRPI+<AKVXrXebaE9xY!hx(4oov%?aXySXxg+NQEx&kzQJcv9}hfZvY@JKKeOqLFbl#<F+sdy
H9*kYGeil_5)<D5RMm@OQ8Ob)V`tvFND~^u?G_s*;J$L#D|RsQM(9NC3V#NA5%)mmON-OD78NMP{f;;r-M&fvQeoLby4F^2wsgxd
VBG(inqciExwQ~A870I6S7q?-4HH5P@`zGMpq$;(!31u-kQ(Bj8<KZmypmOX*nLA97woIPVxkUBsQJv<=9k)&7^no$A5ui&255Tn
9N=SuCj^E+(UUG@rkJd3dfReL_WJI`=+fl2-202p#T6_@$~nS2Lj!69xARBa{USd;OUIvqj~jT1$+3hS!`CRY-iEHLUM<t~h@lEb
)g`}D-%xH$_IwNGz!VE~!LPv4(6@3?>xFWJ2uSc#-}f<+Cmraf@|Io(+l_bUY3acMngFy#@3F!eM04SsPy_aCxI@=nEHU`sPVP^k
-GtUqo7kz=V0S*~Sa0nK(-x;j&2`5oXxa`Ya40(c8BLIoboUM*;3ss@QD<27SqsDmH^_%yipMG66^E~9lqa2{9H#rN@)7Pz+uz?_
b9k2WM7H}yHA5V>dA~L}S#xt;#h~7)Di|3_Q`?`2IBX41b*z!rjwwu}pA7q_k61e^ckct$!ozELf;*;D`a?jGbtq_wBR@3H8MR~F
P|LVu?VekTQLZwLia*P1re-6HkFySQoBIT0@ZHagv&1{u+(i=nZQZ!Hw{}62i`8OmZZJA~htH*SQT1{Z{R3KRtO6rGi&;X+uT!^8
xLT>qjME8|{wa7k$o;osHAPrZgRMilFdnV*MwU5Gs<{fhoF$qd9maX1z;j2cATC`Opr`7Wghk=>yvC3b^Jk+I_?3I&<vKBbyubSl
HB^I*REv7fQ85e}@E;gVp@GY^NIkqGt?`58L6p;ZH@BlFNPlGAVMhc2JKLqmmCSN-*!RPDB1w)(2O9QTylZ+E3M+}{fd*{g#$IvF
_qdr~*g0|s0pi7+3X-?;zdCpgQL^DdsAt@jJeLAlO4AV)>Wa^b!&2w^`S-?q0Z9@#U2yDlM<ElQGGXhZu~tj$Wm1pTzZvJR`udzl
{Ta)ouSs$p-65U>><S89o$4<(Lt*|}B{|^yiWH)CKNwp53mmwSF(`*w&w6^9I%<uxRH4gyZfwnnZcu<6cu&}?q+G12J?Gtblh5q>
RrMFQmy<4ilH80vr+D1Ag+I!0SJN@ocw1J+xht{Hv^bi2N<Xq%)>w-9_Xri(mf<h-b7kzbASaPM$8<MNLXj7us7sWt3r&%nAGN&y
{1y)X>XUCo%w!h5(Z|i!ly(bBH2qSIJfl6s@_VW>SrRP<E^`ve;FU;2XK&BkP%JIYG(WSK)^|XAog=wlwe}55bi_8WiA%@MZ>)+P
OA|dfB@X$A5O0WyvMT}PMft;prSdyDKlrA-Vq3r&`t>SZa>Mc(0}lo_FL>I8F#vc?aU;A1N>Jp=%O3}SX`Na`lh%H@E#d<39TqJ0
?PF$1L66QK8@xMi=4Y0OE?&t^N-I3YhLqg;2(+9Q4mJ<9-gd9&WHR<Q&@jU{Bh+e!_er2ha(ttIBSneA5ulF<AOAy5RYPPBbMc4|
_6|SWfTw?Pe&ZwkOz{MXn&K64*Q(ZBp_s@i9C(R}grrJQytp4@f%+)5Hxp(Ebe|K)d1y8+>oi@w-ooT|1+EJbgpaN2ebd<PfiWYC
$Xf0%cPd-R0ayUv)%$_O)O2~Ht#o=`@2{}Fw@`*mgQrvCvhM#QiYXJ#wsJKn{;o~(U3baGk%_WU<5)-Du5bGPd7ImRclII_AYF)^
TJ}`2_*Z#BQK+1L{AfE(D<q&CD;0|LLaKt7pe$0hA1&_RjrA)_QFLH#9?%upT;QET)+R)9BnP^5zOE5qV*;ASWdBOg@!PSSf@Eu*
y<W8TNTG*@WgJ-r#b57!=6$dXO~OsJMcOlO6gqU4v9(-x@BdDDQ8uSwFYfJF#@`3hZQrF7(xB+NT=jBgrYBp8%>Vue1x3VSDi>9|
wRdvU#;e*$h6%_Lg0}>t4!Mm_d+osEp$ec9uqJ2NzzKnQ`0k~_k0k7>OJ0Q*Hhzf1$yVRjY9Ud=of#HB6tZG?EyIxM(z0#B%D1&j
UaGhc2oOyhD)06I95*vf7m?Kf(NO<0?Hqs*q1~V|@<peF;fj^=Xbj+R+b@v^KynwgV5KU;@cf!yTa}P`#Ky4*Q>BDUMNEkZb)XhY
+(&;EpsRD4fjmz|opnc$g4$hvIC4|a`F*F&{!DlwVX8^d=C6!e@1zffH8(0(Et&8#^LRnRWw)v7XW-+-SpZlFv9a}<2hGva)&QK%
1KXl6Op$QoOm-aH`y$s4Vb_Kmr_Dk_BGpjyB*X6tu8gSF=pS`hXae0gvfTMt<(J0;3pI5fVXSs~A|3gJLe|=lWeq&hIpY2slxQuU
M7D?~9DsIs5#)8<UGeiEc4VHBg~dY^ngRA3T_is?K5Z@pgg86^75q+UAN6YUcS9VOh2^-thD~%gnHKaiT4p4e5P!@q?3WFeD1#)f
ZO_BI&r!BP1qwtnz4nU<sIZAP$X+gp;ns7PS5(DVg6@_<{oif=47q89z~26lh9vXjg<dj3Sw8VBPHV*GN89z{+dJ3W8+k;|N9qnN
k+Yc!yK4VbzrQ;)`>vv+L|<U>+9hjt7PsB&B<$uTU+FQKXePuZxTYbCrQQG94Gb{Oe0|@(6doOf+F_OkwSc+sbN_xNvTXxBRoH=P
Ids#$m!S(DyM(F%^O{Z^_|=thJgkqOJHgFY4#D=#4|?+6C)XpfmoxC;7Dd3@?<QX)G5!~lkAF}aT4?RY(sz<c3SSF4r0vpewZqyf
#;wI}FS>Zjc(*%KItfKSES@syiGbx7cW>OCH7*~qr>SvgLxJ%TFY;HcHtE~<L8xUFptN~7o)G)QhVc#S^(hSo53L$}mLJ|g`3l`3
tdwVBB>suXwej#SIrMC?4L1cbaWudWfNr>;uYP}i#Alz-_}AxHe!R!sjGAG7;UD&9771CQ=2Dku%U4Fcx2lAcGytiUl{f$gAQ^6y
<p!hfoyw%Z(N1#q+1Isis5qisYTyH+=TQU6EqZ(VW;xRurqS<Zr;-tV4_8`uJcdUvu{h&*&~KW-H-g@#ZOpfGe3s7MO59AuvS-9Y
%$X}srHP|{^H*ubr(aYyqTm0v2@%%!vv^8dI#Kw5YIzD-^qS*N6pO8EZ{wSvO-?Gz|C01s+POy>I;p@oOejFJm$&ot6TH)0n8fHu
N!?ZZT3IBP!xSAIhQxF-buA>a$?DPTA{#XRQg+3L75?#luKgnxKm(1o7-9O3C7V_=s;V)KqxvX&jU_Y_WsF!!1stsB=W7F2Rq9#7
k|d(0JxIz+m$d(3&z`s$AGcB=puDRxp1Zj*BKUcJj#%A_1)>DB6_6!cXGX%s3Z9I`caC^*NBg8UWDOJpAP|rbGivK>ZHGNei70?R
S=4-HnG#kev+1L|pCWD;+_}zhHk{rfB5p?ta@Q7OeTrBN>QNKzAEbTu)0^CK8gY0M1QIpNW^AAw#mG{-agE6+RUFVj!XK(D_U%W1
Zm6k(h`cW{Bx`7xvGxxKMFl6<T$LYe48OXS?=hB*q$h5ISxDDrDtCRCh%JxTb9Znb^;Xw+8y7BEG{DutUnnl+^6cY~v%{XBLlA^6
xqJ1+LxW3s3J48Er<`9;nz6|%t3xF)vOl1e8dYAeG(Yz+CK{aq3>l6_uJ&VHD6tmdWh>f^ykKfG7nIGNDEok$Zy<5jQUsr4NVW}N
ggd8p1!IX1_1gYY4LyDMl0>Oi*l?Yv9i7v2F?*?=QQ%0cRocYuFoy=V(-}(snvW}Q@kyzKl`P^!d=peK*J3Eb>MQXqzaoHqH7Ic~
qn-X`L6%yWB&&S$uFaV7!$#O<P-5h+)&}1L3K}IlwvLViDjEGufl5bCHb{GH7a~$0X^4i4z^h`y{v%Ph-WOO;RW`)JGuI{U7q|)E
5ogB~P@sCqoZ1IiozPeXd$~s6K<c#F9mS%mv;rp}Qlv{vG08=Ox$!Q*=031Qx7wmXq5opG+T6Q6rFTK3dg50#9)iFU{N#?bRYa>C
@cczAgUSRfT9OM;*=5vQApU@ty#G36HY^?;%GCiIXcp_Dx3PLEXB?I;^Cr$-=i^QwLSOc6ulI@vfME6|lDbY%wvF7T31QwXWZh)@
zod_{tA}@;lpXX8)cySf>8qmHnYI9IL$<SY?_Z?#mpaOugKj(k&Bv=}b9{(Hj&5aB;Gs2#y=ygA=H?jhIHH{U9$YkYdp#ZPrgI2<
iYo&&#WlMKulQ3H%_aBLr7Y9jHt+COZ0EK>XxzLl4#0_CFv0)@|JhVal54hsnZNr#lRb$Ck3_h~{{QbMu@X$A@wp@Dxz4UYCkbc-
14-pWNsZpw4KYxSf|NmhcIjz$)Z_Do$Q0_2hn`hv&FF~DrM<<g|BeN%LUtTpNQ5bu@}_ZgIpqx^vb+x1_M`J)tZi3-g}##<erNJY
mHjq~$@IWX!@sT(aCp$ZBuGb3o6>nGQc(smteMP!2_^ssA-M%7ThB+yKbv+|`ZrTr@=G2^c<#)HuM9^eHQ~Qt*HkXiF@z2il5nRk
Pe`h47Y#i*sh(J_X3JY9hTy1Xdy)Y-wxtZy?d6(vsX-T)kx%e=(Rhe~OV8OwK3Yuz%f0;y3iZpzGOE+Md^AkG86Ms!B%FTx6BoVK
eDH{Tx)8%NF*@E`&}9)B8$GaT;TMK@!1$Dr(@|SSq=zOCgC&Uh53`c7>OZ0jhbAg4P-A@q?QE4R?-N(9E&hf64}8pfQEi2T8cQeP
Y6(S@JCjV?P;grhg|_N=*#6;c?-@W2ZmehjPX|>A(`YWv_Qld09@#~B5BNN!H(rw!CTayO^j8iYn2q9i$bMA*xX9M&A9>LDAs*~@
v{Bco+uW+r&^&8h>rMLob*_>}&^>X^@fMABoW>7v*2CTU&MXCC!kuafmb2C(rX{t%U0{u2y(<(g#A@Q-|9?_aa|-DKlPJU)ME^$S
V2QwkUJ58Je6uTP04yVTD)x$}*PP_sCKyADX%h?ZlxHiFD>5c@+*OVnhplpw#fIvk#@#kkb}DkA#KHz$$=WOBnS0tISt~%Uh1`lY
jv^7cL0L~hAP2J?zF!Pls;uur9BLE4jv?rRsw;owWLghrtJrUH0N%uw={S%RHDxg>Su3y#LYA`&iPbL3{t3*|jG%NUz+EBuW|ING
rOQZY^cgA>j{zPSXv#hre8xhUzQlOCnQx_FU#R6@Z_Xg5t(UgyER4(YWP<^OMv5~TOwQ;?vVLpjJeidMHXJFa=HGg3SS*K7KSIt*
{YUI_c%XUOl-mK*#^=k#6}LOQE#=__Kb0-px@;u&6*mYMc30Jt0h@$Sk~^bPD0R4g^emBYR2|+^V7Ozm8idJ3uYziJ4aIEBE!VXH
Xc|%`z*W@*aq8rH*ky8;AU>fZw1ArsB{0CZaJLI`XFMR8^eNKzJ;(Z-RdZJ<1z~F5$}0k|AG-B}wLk;=?v9$3BsX;%LQCM3KgA4`
`ZIQql#zPnWLj>Gf=*v5E?O3!Z9i)=IYhO(+cR0A?*4S@pZA+?R|$RgG&oH+NHUm%PKz8r?1+g>Kdc?}f^+<3to^J#WEX|8Cc;j6
HUsn`+PD|PY?JERB}*%Ev6uE0u?ZGDZso3PryiB7YiQley1N@)A5gT^#k@0KCCH(^?_}=k+7E|u*{6t7HiK-pBp)T8T{}xf1B0KB
TW7yE=PqoGHrgg!)aimP7qgH=HwqJK0H3;<?b`U;``_kQWnHVug+MQ@NgXIU4}x)QnGyv0M{PKxp<1LAf@rD}7-QDW;i4md<An&k
dXi+-QG2M4z0b@&D@1nrncbtl+8I`9!Owe^Y_kpzx=1MUlHTf>BUk3?qCs9qSeBVE@%W`78<W<(r-JC!K>+_-cZ5K{@c5ezDfh-E
`sRM2b)8cD6viI@oI<n8$`Zx|Wh6gsO~Le%rRqw2g$f420j1K^)wOHEL!kQZ)ldMzS_B|FIL~fvgMDYvAEC$CNu96Y+d~D@f?dih
Ae<VqYH_l(;7hx|Wb(O>ZEr6sApXyfa7PP2*F0XAXUz#B`0Mz^?7?0w1~`g(_2MSDA-?V>zVQLws~OxyzpKrldDd=Ihe3WMqeNb3
oR^7oX22XuUVGFyA!^TvP$}R8dSP09{Fy7>f*qhG{|}t(zST>@Ne!z5z3&9u`*jMVTbl)m<49wuWd7>2aJNpXe_9L&2~L`BYROVh
Uc)iVham9mv1yYk3pV5uqOQI#G+U|alKGW`iXROcvJ@&L+9`xZOmF^b64!cqdc43M9`8C{Fy5g&b2@D(Yr7&VtzwhC6&v2L&~qw;
{V^i@<PmVCC$CnnI%g_-MAK1b-b)=`<Mss^nbZQj-S72E7qdepXH3RcD{elmCDJ|$NN3X)TF{X^W4`s*DozTCrLJ?KyD++Ew^}^o
`9hfFF_##EXbg7GI1fV6DU}dQ;9O@Hf^jWDdzjVYejF_ygQcFr=-BEm)(Drdfk7XJ*})8g_ld-p5`&na+@sp=V)AaX)Ry<x*$Sed
3H$j>3l=n0BD1{v*h+r?a^KJP%(tN7kok5XC-K&kD3lQ&vu(Bai@S!HQqwdVn4iJ;7N$$fPzLjkzX;M1|H4DQPGL3~$lpa`{lt(a
1J=*O^|cHBuH}c)C=D^`TXnEXFq&lE*~T3kpff;5v@JwAWXpi(n|k#@GVjzcU`y^Ylxliiwae#+<ucMZg-C*#^u;RUWJ#Wr6?aKj
@dglOjD_D|J`-P<#K(J6KOiy-2&y9m-Qg8)fg|&2MeVLMSR9R;8mXKnD(~)L7WUk=K|HIc!NX~fwU8#+H12yT1SB&Y2(iBn5+sX<
lRwl9IfSd<ZQmrfr8AUJ4t{4)hK5BdGu{`3sTpPP`6)_DXn@8-0mop+p(XVfB{ew2S;|>Nu%9YiALFm4)q{<agBC@hH=5vf{MRZz
Y?|Axbw0**9L}e=t2<X2KsnIN7=KB7J!8=TU1!HO8C*XIW+GS7VR<U~@<IuZ_P7Ogzs!qw9UTKczmlIgzE|o`?8nrEOtkX|_0Z!f
G8Ccu5r<WEA^mYA$?9j#v~s`xT&&twoBVw7AR0PX9G-JA*6L^FWV2g_^Y~1Em!YT11wRNhB`J~3vQRmGn2!1+;(D5j1fYGJ3qv`#
MA^3O&q1M|KSr;&;Y=d=7E|QE^X8mUJ{YK?RgAFg$!0m>#QzI+1oZ32%j(m5ED`|NtOLhSMG>zJ4<f8WVJqP#M(1T*rd<9_2X9D&
l;sR)suM!MB*g8rd(QXEJVDp^yfK6R1tao)e0)6=G(4-+)l!6mHk6s(6KwOK^ks)Y(55T8<WS_2FuIiVCY<ep;8?wfZOvAW>a=Ut
C}I2&VKLcfLGZ4b0%$O8UP%&&<GP{}*M*Am<D#A%@cxH!JEPbLLh^sC7pBPIrBMlRQGE(SvlD#|Io4-t%jJWC=DM>%(mQU9PzLWZ
cGsDb0kQKO;^?OnTCPgWwxt|iE#CZN2}Isb-pJ}*B}d57${X5DyV-+3f;gBK3lpZ{*!b3eJ1r>g0^H=*z<VW<BPaySua!N}A46+l
Tx$@uDMR%_u$2HVC#ri4I0`_vER0j3bN1hh5C%TII?W-;xLw;aKpx64HX8F{ED~G}K6<^)r%bg+OO*?4ftL?V{^5Pi4mw-#=^Ag~
0lNDyKo!Mad6mIlL=typ*Hx=j-42%ae78P=kGv{Vl$*-QU@+3xsm>~@A+@n*AXq)E8Gam|n>q{YZXGvekAvyA$0R#8zh{63Awhxj
55t8@PO{#6vpj%2M6!-Ig-#{-+8aq>{^*B`wvqe{-9lpjc>L23BH4ppotkDU6H~fc1}0ie3(-imoDM`};&xO9Hj!J}=AKgE&EQZ3
Yc9PGMdAMEg#f?MxG@LZ_+*=LPZp?6A}yA~@G*ObHLZC~0sbB%kE3tH;m~#Q(DHBJ*{(a4Jj|?K!5{N)+#~#zs%2NNWnYYzmwlQg
{v`vp>~(zhe5TNKFMn&Wo9sg9Rn>enImU<n#YLi_8MQ=d(Z-(+i+5hHi$|$Ns1U)6D2Ev{M+G1*N%NGuG2PKO1FPQw28DW>{=Eme
+&YrCL^qSED6P7xP-V*kA`tI7%hOIp8S_O8U9To<r4^zw;ripfot=uvuEwqEr2*xmmGx+~>c1JYNksk!02rvSA^gh&J}>|keFlg8
5iTZiewZP;Rn)^pyD0|Pq|*}WZkb17#eAc#YgTx=7(qzx0{oFM$G}@-&vLd}TYI$2q@Hx}$Z|^BQuh_j9^=W#4S4PgPW53Q=9Edt
MF3V(B$E*#cN$?71A}d=)cdMF4N>D)Z3pnC=$Fq7yNIR5%CjX|^{KPh`Z=x?#leuhroEkgH?asU^L&T@L&`Gf(B@3}-h&-iJqQ(r
W9j;Y>Ws4)wkuV1;B6Slf?=}D{|*`e-BkH6$x%B{vVUu7g?5&5NX(u^wKiH15~Z1ggVc8>(;{D~N|1wq3-sLuirIC^$hf|k7^2Qe
CP!cO)W&~=J+r|~g3=+f$OKpgyP#LNcKBmNKcew((Mh6uHV2!@J<A>!>!4aP9H7#UCgW2LDzLf>8k;}T7or6mtN(+YBvSrl8nqVb
*~=@v7QCvmDmQ0CcAiriQw23elb77T5D9m+1>A|vjxTIfbV|RlQqdSg=1g}WZs3)y-y{VR0Y6D&mB>#n-rwT%{@lWmJg>*5>u#-X
z-&$iENz>wQeDhU-c^)$Oh9X%c6plSFU759nqRAc){Lk=_c;0@WJExhY?U}QBi%!eGR{v#cqU5@$VbkuvZdrfA5)`)j61DReIc;d
f6=cg>Izw{aT|ScbHKVDJZi_ZV|0vrkEO8|^<uSb7nU19W>Ja(?bz1ed6o5d?Fp9s1*}YwD67KoGa?jFLHr^KsQ7KGVK1_)gG0g)
9ZDF{+H>C@OIo*I6vxt<Sq%w|m0_GGyvVmh>?e!>8<{QN$^R9p=<9?){2kfnQ@(e_Ta|dNggO@0)B&V{qQTB!VVmCYd(-6yY2SP1
|Fx62R9TaTmZL(f9|la|hJtT>=s9k2f&VZ`sYHUuz)dpM(zz9EdQix-L9N~oDTKzFdD9`-%q5h_?uaO(!E(2AH7o$PCh|WP?q8uq
dyEn<JLQ<ktcImSpXyH-vSu>mgU`8Z5;uKR|Ga)79U3z$Z3yEsc3CTe)nCY*eci;npul^<vo66zWtbfW`ks@IAu6nnTE+Fw2Go2i
*}DPhUKFA2knsZfN`7?{6whK;spxcx<UpszSUzZ{fNY3mlFbsPl3ZiQLmgfu#||$_to<B3ovcn;Ud}G50SRr@n`0wibVu_*yV<md
Go#X$8qKS)c*B4re9}yE6j)-S>t6D8Ez-d<HX*yZN$ymRS1~@VVd&1>0uLC65`Z)z@q%Di<B%%NJ@@==;3~#ydj8!$XXjHQC>=XW
r^n9jm>7}-^~)m`5ICLBdzBepKU#2Io?cEm)llLjcIs>aDUz!vO>*s1I_!ubOEXj?2>Q$_dt@8osbEDE<7R=h9c2OCIZ9Pd*x+dL
A?%ULJ1m4mdxaR{|5>4CqBIVa8?iM6;U3P&J-K}UeW<M`07Dr_lk+xj5c+rO!@h{5{@C0fLLiJ{t$D8)7${NjY<Y~f^W(GOSlp$#
1yjad!dKF@lqY)Ij+FR_JD9f#cKs#aX)?bP0HvTGa|r$qbAfU^Q`%h~InWyYGu>=`YdYR$Uoim`G`yOPUysOga7zdVLvN${cDtO>
Ts1NfkRk|?2k~$^iJxeOl*L<$wFq8;9g98Vi2ab9WZ4yq+Iy7pw=QbhJ7F$eB#drR?L2OI7}Iv3Ahh}U*b;EDJgZ=vSB!i>6EaP+
_HOgOu@dg-%$;V7h|Uz})WoY-vO(uCVy`73m8*{WqUOgW^|z}G%%T+3t7gKkm6{%<5F3f%;$RQCuwyIC*`q}2{V(77`23rSM<^6`
jNYNU<>0tp_t8=1VSbxbJ`aT#BuiadHCuF!^gAI!FS#&}Xm)g`P%*gRyhw>=msuu>hr6TRZNop>ylrkb#*|K|p(B*|4sj}sZO>4D
3)bvy+@0pZU)ZUBmS>3n(Ha!QtS49;#qj(Q)@2HSG5ndv)U})~UH!9mJRF${b@)z^exND1TnX65zgR3+0bSh5R2;X+<Gv{R-sTMO
0h2$tq-`I%ZK_-Mqp1-GdA}c2J0tJJsDjEs|MQ@!KZ3}!xOI0f`8X5ceFi(9m5)c*mE$;JDJt|~pHNk_7_q*lQ<+cAl!fEQ=uh|a
SW2iEEdJIkLay?uXTn|2lngpLRW9@Jol>^YE!WL455H=?iFu#@NaNy)ZQPF{KJl~Z+i5A1<h3Kn|EWHY$|fqz+w;XsveY#QiJ#Rv
<rAG|L2YS8c@3QCK^*WH+8nhjr8_3=V;0jtzr${xc?428`heb;D^Oy6#!fz25&&_@d47JLbbX3j)&v&gC+vi$Cw3agxjk`u&9#+=
xaciwy1NdvZ4cBi;GEvAVX_n`yld^M=S_bM@T&{jyGX%fOHiG;fs@m&c6_ndHLWsKiC5hVZa@c#SGREgeNG?6KDVB_u$Mo{0MYEB
^GePigF*(f9zpomuTgWcO|VF*NVxdV97R#5;}SpmI~ZX&2<g?iR86<-NdMujzkiIe43(Gv8-!FgPd{iDV+u3eJ-IBGk(btzjM!{1
HVWjxK5nd`89A~|jascqLG<a2eK~P(mj@?8uAx1;R&<2spFnRyE47?4;xx$o2ybj3wbXi*{qswtQUj=9WndGZw>mn83TY!wCi|PM
0=ZBMxB6_fIc)@+IP%<j!M)X8R4d34MgB8gXbzQnDrn1M6rQTUNp(dtekn@*Waw@9tOx^_dvuEN!7B#U(0yvV2RpmDni-T;@Bnr|
k4OKvhX4LNRr>uTdK^y4WVt%l<>f@PdQE=&XVUL!s_aiJQPHXD*`d<R+ZycfIeYZ?`x5~d3>VNMHFgJ9#!Fo0ZtdO-esW-cy7R&W
I~3}S#Wa&|4M4!ic2qUAUOayZe{r(+TaCPtUC6hi$m%};e)WFmkD;(S`Co@vJ#1(2T45l%Yq+)|J<feI6Vd=Q!?gNF{fdByT@He&
Z0z?3gNiRcu#<Wn6*r!G^jsejPmC$!7W?fD0?Mc|0elMS{b7(EqA~uT=!=p&jZ|8)ZUADrq59arGn>RQ17TQ?RWYqH;ad<r0~Avy
sqqK)3PjWyLpBVlDUh;#H9iQLSlmkhila3M{wT-V@WuGiO7y@|+Iw;PCXJmkNwE9&3ry$qo%Gr5v?CG%a34LLCbMW~%mJC#l9NTi
{Oj1<dBMkXr;ZY0@^4S^Dg^&UiE$n`0M&Ols{cVxS*?b3SN6quFb=O;uU(ePCi<;hg?7kT&#5oj9%fiY{9?gLUkKLE6@r<=+#+NU
*jXVTQkc%&!*R4VIO{cg*?&*V<6QFsTS2AS!@!P&u~pbkA%cPqMGRPzhj0tP+$&$^d=n$c)!TEQpmwD4zYk0hx&a}?*sP2ZgDe}+
h8EGTF|EMkh!{CK8nheIc+1rt;{Mbj)-=>Dx76UA;Pr{UrCQUs=ETPuexZ$Ko(%`<Y%w6$_x6~vUsdxX#_P|9l&{<egv}z+*)w6(
1I(f{)8&CvkCl-jeq`(XUu`b8-tIoA{WR_94>aEHw5-b<f8Sx)nzL(AHKbsgNEE~W^kA@Q8<;gxhdgqSD>(TrfQkQCeGAV0y{2hr
Nr$gBxyV5jlZ3mmisbmuawq@DxWWBp{Kj*HBu^nv3dNQRay+TRk}|<OEps%uJfqC!W_TQ(uXJ^Aj?Qj=V8JOJDKGG6iAEv$gbF>B
f5+1&y$7KtW|R<;=gx7}P9D{kGcUmA)}Ve%Ux>l??=w?@HON*vvpvF(*6-HbtLD;0H0ku0wT4)ue;ON+bG2Y<LW6_M=cn`GDwEK(
yJ`tQI|e_EU%4bNT?kJG1Ez&gs-QecLDK0%UbK>|68ltk`KsIcih^p)bt`xHG=?-lkz-UG>zytlJ^1#>3?|pid*n&m;)r{)b^Q`v
sA@A1&<*|(03NP9<3MQRZLy%8WaLpy>iViL1@42ArGF+-?MJkWQ7DCmeJPwVpo_ZgMB(&}t3eK4R9^!^H6$AI$+yeE%XIbgrnE{0
YS=IGr*r__fK1(>6hJPt9vQO{)ciSq8b{l@mLzvI;@D5Xi-Ybl21i|{w%7N`2w-aX%6Y60-mQL$K5KWxa-X&(cUHG!o84f%Bwc)t
ublPvy|SAsWg1aCgAOi~ZGqun*jy?;2muk~x%Da&_Ftt_ex)YA?e0Kp>l(<(bE!7d+KCp_smB*p=Aq?UQO{StB$(;!)eg>nKuXqC
|CGu$L+L+<9l5Xcuo1qO=o8o@q;1}=CdUtmuOX$_q~xOE&zE0oX8p<f%k`*{C=q`^Lg>}>t1yF|ddJ0M#_B5kXkzqGURhYdIKWr3
4G_E)bET1H;Bbt4NOqT{PE1d+{+@$U9m65Rv+D>RBq3$C)hviFNl93(=rabo&)0eMW6mRKHSt)=!>*s>M3tMm-#g*#KEUfbUYM_T
a&k+BhSPh_EY<+>5tdqPp4|USQP+WBA7Yqoa9G=%!`K`8zNGHyRnG7S#Qe^sA$D3<2Ier(zX2(KnNTxv6dYxfYzM=3t7ZJ|^1<Dy
TbkEB$o8n3B`zq7vCT88)+JY-hB$LrS?~YiFk9Ho{KhxxCcLYf9OisJzDhrQX;K6im=ya$Q!dxP4%m+LP$)POCBA$-WrKrJ!phMs
z<8XsO*>8*+h!G?6GWgP1}KMpSlFhdmWbbNWH*{Hc}c>1Q(gl-ggP0iO}@=C3<kiW6JksRe4KRa<ig*UOXT9_`=iO{;?|}3c5%~_
MOMa#pvU_&#O?3GYqJ3pD?{pvS2%EINR~I=3NPvfL=<axxH!8<2d6ihFdId`r^x@#y<xYqXMh>F2eQn~j)pSaMm=$sbqHrgCcXpc
I)-OZ6wc~5|3wKd)-&20I}KqJ`K~BF(hu<7ig%}1!Vmn2nqx0`JXyc*QW(9#89*Mp9*w0|$m=s_cZs=1T@cBo(=m7iBuT>V?eyAT
grQTI=4H?Rc9zD$9xl@OZVxymRVy4FoJ<6UVqmC7tc~PVvL@u(#B^Tv6(id~6<Bdd(OxhIm`C4oi)ewxi7WAHNl6kXh$FOO-hu0q
0_3f%S_MvZcWj?Mh-qma#=md=GgqLAbe+B9Gb1l17?#+-IgG!V0~^=76QA2{Fq$LHX^E?lit1`R`m9t3lL(;@G5!-380kEo=vsOf
Zbzr*LwI!&^*DqZNLu%fmvR;<&46_gHLETt7)rLcZxn3qYmDEZ9VqEQAD%9pwa}T05-k4w?e3f<k#r2qEGqs}UnLO((_<m!KmM0r
uTR$AeRmi)1{SpvaG<8U=UL8XI2sWdVIoh+AemQPhO<zWfDVs6={CZx>n22TSPTd~df|vv(3OA6jRS`&hI)BS7%HTBRuHpoxDV7W
B9A1Mkms^5W(8o2cls?mdz-!pOk$6}&mDGk=lU-26+_$k?YKm}T;ararUQl8m!^;^(J~UUfIhYRziDibEz5W{>@jM+ptELBcU<TK
_vC}Eg*eGAIL1T^z)Z7)N0!oipw%SPUq;&A2%Q+=ZK$(9OTxT~2)eGvnUibZ^i%DO$29d7U@JE?&(UZnnw$-WG0n|Z=sUgA6WQbR
@I2R6=wk>&0p4XK8G|y(@EVwt1O|&a@oKaC0wL#0c)xu%|E!BRxkVm#1!An3W0{r}ItBg^UU}EV_7N!7q$+7=D1~HHLF^AQ-nXBu
gi+pbe8LIWkIq?GAB+Zx)bt6st)QI7?>n{S1;=>jH2(PY+$U;4Yh%Z$X`AQa2*cSB+IG^ixJJfZ1La!00L_?raCTovagFYjK*HnL
dqUbP60C;MZ@0BvvrE==`T{8<-&vBSy_3ZJz<C3c&kJ7>!KxnY=u@!WcE%moRXH<;-ba!-DZymY#k#$g;@?_~B646c%aXI`e*Um-
ac_gmD)y|c$=#O1W5tae6?OfA=g6Z{sK6?7X}9c7@(jIA!*-JzVmL5M#wV3{%?T0PfsHbB20W#ntwqY=-3iL)2vzK(8PO}FX-%H_
9O?RaS$vev8xKB%yCJTJhx*ie^!m9FoWo>!ZYA9tqNG><d`Ck|Vg-qK#bvNcgPs@+p0XPG54Emx2v{@AtWv8P3Y5Jra>RUF&IOzE
huP50voX^c$Y(Xk*~&`hLI>Pzaa$Pea-ElsG(J2+p`Q<c$q><Yg9*UIeP6vz<^bZ9nP1db-H0+aQ;P5tB5z_2_Bm~y-T^>l1c|GV
YICcDOy2|Q+hpMgSZNy$!7OhdCv+aR;Rb&<Kr~;QU&wK^-;n&Wwk&wvke-Vc082T{I;*9&!J)W;0FknlrF9UD2$MFEW_VA)yvi1s
nE{7(jHIY7P!btCm_=*AtJ~Vfgnkg7>ekm+q3|P6TE55OMTaMRtENxXCylEHrsXP9(oyFzDP!z7ifQ#vA8@efe=18L3_V6v8s4?(
&TNz%_6t1q=Wjo1LT)D)l`Uup$}so%Iqx*RYTS_yH@y3N*LVQJe=EKMA|_1mFGy3;Eq+6QoQePLJW3)b<KRLiw6Z+yaNQDlsyv~H
^5y|nlTEmPGC5M<+ky(~=JbD`ndiO|2_yR=ab~%x=)}RQ2y*31O1TP=9wog*t3+0!5`~$P3b{hOK)Wp54faxU4l3+OG}8SHW_24G
!+fg&Wlc&501il}9<-*WjfaK#3Zw+Ra#qb*jqS%1j5beWRi{}2dMJ|oVUuYQ?N6N9L$iXi9KZIRQ|Bdx6k~=q#g%V$s<;iFYS6?<
9?`I1KaSK?me9UJA&i*2T3|t}=lNm}5bct%3t|^z14Pxi1ot?<tcL~yMek)<6}tmmj@ADX2u+pA#f_1p(#AkT2t~)U>@T2<R@Wkx
hPz$_W&Z_6`mUsJ6iAqmJKDVRt`hW&f@G-YZ<N>?-`~F^3^)p5rf8=60!Os-^EBdda(f$|_oiqB_}@DUp(k*a;g~jc(26?qfST3F
5&4MwZGxb<*KZumAKNFJx?27Ka;Yrp4|5@<5Hi_|uqr^BQ}NI5Ofo8P9(Ug3jj_kTFemcU6%?>WQ%F8R96*Xe)huZDT|L6tCWLi$
qN(8_59iyn1S>>!Ez7b?(lSus-pg#AA8W$UJIK*Dn{d+v-(PjdRFgeJ@U4v;7f%34HTyMvUy5OiL6JTErKNbxEqY*g%HAno#9u6(
15}YO9T5Ccr=lIE(F$sGI6O|OXW{pA5(P^P<P#Snr#X{=-CYOHO>F`YRUb-4iD$x5i^jE@WD;n&-w5p<?D6fKvNxyWh`Q8wePV$T
8!nVqW5S-o<%T84if%Z^dEm8i{}=XT-Ks&f-dt?Nh>zaS;I+sLIQN>`5TQlg-<Hb6-b@mdU0%1`D=-hIib^o&z%$J4?96h{F7vZA
J8k^<_r$sxCyjlph^rpCgE7TD001i&C`st5vWrtg@)Ydom_rgl`i;iZ;ftvEEQ(;9RHP2`3njZykQ*_&JW0<Nt)X3dOhK#Tf_hYS
v0(6*&PuFf?@Qj~DeVHx&B?$7Vt0NkY}p>=C!Jy6k_{Cl_>(UN>}}Ra9zY+$ECvu%v^#xeXr12c7|LlZ4pTK6#>j?}&_SZeImW^u
xoTuc{qSb%^&!0k2UJODCqUz$<}oTs-BNfb+_Tu!ew}k+($$6zPHbiNAm?6g6ix?i(+hb#_?>3577G7+Oz&64RC|AP;TZuLlCv8D
Z}JJ|)epJ%6sdYT)w%#n&m+RhO&7}{W+lB!DQZ$CDi@3|I3w&E#LB^t6^|Oi1+%8d*ehONLYkG4;r%>+H#)(@wBVsqRyZWOCO_A_
h>b<hB(IVs%gG1>f_AY;bnn0^ed<QM^bIR9jM{l@f+e_7_k^!=h5`Je&h1jlPdOZ52M<r&yGQJEbbb+kz~(Y>dc^+*6<`md-n8K&
A<jCMZq!uOG@Xql<`KvZ<MweGN4on*B>PHVnsezOR}&F|_uG}&K0X(mYz)f=lU!un?5g9#L1AC<70Q?!nO>f{9QVl0Y*!X!=Qu!8
33lbCsK79V%7yYRJkVhqXRJiG#hft{N&H{<J<WtR3jygR{F$h3&`?OpaE`zk71xMdADkCrIV=*_DLcbdrEj$iqc;Ci@21>vN~t0Z
b5usjy1>6pu;L0`Koe=pr{<fg52|+QdVo@NP-Tdu%wTXue=+vcghLE%)nUbHFmlt|hL_DxQgD3XB^;$6vydiC+7B~SwcS{LFuw#p
ipCRw4K(2ObaOhv<e}Zly+NtuEZnWxC)^NZErifWL?B@aWJ1~@W#ozt*G75~U5X<<NTn^d>ft~{<>U6*g4Mb0Q->?`aU$%783DKe
!7lyaxe!AlD&6%cGH)hmqupbe<n2f2D#oAp7j%m87fMKd19l#MPyf?vTvu5}sBdvr$R`7>-0ruTEbS#mLOq6bhkQ1c|4Ee?Gw+D7
rHb^Pwv9eRtc@xW!Htepz(h{YKvSqx9h7l+bH7L;P^@l9z##$!7I2&4Q1mm$vH3uP9<w+j4U2Oo2N>g=8okl1zNZN)K?#892zp$z
-&l58HDxiEG@TRakf4XAp*UzYx*@JeDqEY8)Lk5hnDj9<N<DoU+@<Xh7SGTFOdafJAnx$UqA&~w2e@Fmkc%o4fn)MmE2?M!wq0eO
HJIE3*^yz~3r;gue&#mb-`2hkdbFxVkX+PDui|ZC>p^=iD7Z6JwOIA2_wE){4e)A~=ZfbI!|p4RpJjhaze_TQ-k?iBmljJmkG6fx
8aJ4coP~SDTQA>g;Wy3_VHMjCto6NR$=njq6|XFX0k=@}I3S%0Fm!1xX;J8V42X{1!O+5fMV93xfU$RQ!D-;|sNEPIZ=@{?i1b6e
rcl?jSaNi`_r27~8Jy7ro$Fx+K4DRuZ~xnB{Z%S0q$r57oOB&F;w>T)0z<DI7Gsjka}OMwr^W~<&28JrBk>4E>$tN1+L=1wLw>wT
#tm_|6Y+6HcRw9Sb?%KZnlnVLhP0;m5K`%OH`2*v%sZ9TOG((RU7AGp_6G-*2^fvCG_!giIXa@(r2{|({qFzlEmk7TDIQ5#u$4r(
Y9#nHH8K6HK~vNUU$u69DCyqhum}H8Q~>fnTsLPkc9##Pvl=<8<x%pDyDovV$Jsj?sWM^6;&A~QXdq*C$IBzA&(t;23-8IY%cmzx
H5LT(OG>L|f#}&MuV76VWBGo?Gc5qYHD}}B>h3I^1(Gqo9gJF4=GbL=R}cmC{Q-1HNnb&VIOlqwgHmvS$si3_6`+AlVjX({%l2IA
DlRIjG4rAR3XACscA=OL6-G<xC>nl<I-xN~MG}wG_ZzVpL%y9jm5;BuzuuP*ggXtNuIJG(6|U#(<eI(;+InO>KW!~vZyQ{yS5^_0
RX73v?Y%1eI6d}P>Utyc;TKV<%%+AB)wr=qm=<(WOQ+DvN`$oVmBn)*$8GjuIyGJE^2akbZikxbzsxGJ!X*Q6{5qQ<NPQm?*&!Qc
>IoJpi&1RRy1;KtFtHatc0r=`wTvuNi}Gm-CiTP>V0V(Cx~Qa^$nP^qORnAhnSjK(9UKV@X4YRaZBE61e!aMHJj5UX{a+B2ib0KS
Xnan+Oj~AgV&LJU8H<l?D!9~r>ytjY)K3Pa<=}3MpFv9f{%XXSnU`D4RW`;AOz=fwSikiY``9n?n6sqf;^js5GU6ZJFxHXZ=F%cx
Kd4azWNabESK7|}1<0X+fA~|S*eJ&AGc^l$)bUMC=ECx2!sJSzXm`SZMXt#}w+iXvxsk4-ed!hA0YM#qKLte%n7t~;mili|qm`58
f3c01q{P+y+fB=Y91d5+$XAJk@OKHBMbH|dZ{A3Evcz388Wr=E=pAn1?3as*-wiMeqtsH(MoAB*hCzcv_TNwfi8ulc>qG$9c0ROw
Q~hntsjTBuEleR_+V@Rb4z4#;YF?d0*fbjmLv1xkA1(#9=;3&~T?K~+jF=9?Ay16t!<lenBD|aP`KiJ=TZ`ocLl;(+j|Pxtj$-cv
U3IJvw9eVWo1-esQFUz0gn~$A3aMA$nF!DcX7Qd}P`pZS?bf2<o6uz^1FzaC#$fZbDQLL7*aavcY|y-2iyxvBf;;0_z8R*-^*ns;
tyuPjOV(qekKPc8zs)wRz5)pxhl>?%GCADpZ8MJ4xd5=8kz!TPC=g_%oUM~u=W;^Q;F3A7X<GUZ5-n6=1nw)89EzGglLdtcQS&}4
$_LwWeQ+Waw!q2MK;K(hd+2fHIWB0e!gBaNAwEyO!^>-G-h>oLN+|ugG)0BI0<Ur%qr$|swMKg2j1Vq!Mx@6Lu8DLlh9b6`f$|XR
e;0eP4H-OP%4aJ{FPzOCe5w)4>)E!+H`<>2P0f{zB2`WPu&@z7tMK+CIP#_w^?+AZ?IrQ8f#l`hA8=Q4>Vr~>8ul6baa0DS(g{!G
40eDPYz>fnyO434Vjt@XIW1<a6~;SSuMq;W0CIMgghBTh7}5gG%LX~G%VZyopvdCuJwAZ@IMB5OlCbpPxN1gp%zvwc?-`KDfM|8B
1`6DX!)xu}7;~ad{C3-<nJ<1A6DrRAhrl?NAsDaG8`O1G-SIIJ=_gQOMTxc=@g3<o^32cMTR($LI2Lsd(iT#(+xY}H*P*!@qzu*{
Y+Li#VkjKYf6h4?8v&7E#ZLt=c>cq6y!Qh`q%FB`to$q);s)x8r4uR)V(TWu79&21`}5UkHKT^!DqL9Yr^jCekO$I4nbi$PMoC1&
90oD;xBQW=v!x^8WWK$1-g<qR=GauM>am6YChJK2=qj8`HJgoOy<~w6?T2vI*y)epo6fh^@<^sdsn+>Z`xpuu7yAi1e!LsHwNu=t
tr-+^0^py+;s#Fl=5RQoz-NfSdk~RnVlxty>X+wQHA{6}|5<!pn89bFMT0^}dc<T4B1<0rBe^d0gOb=YFYT>M3FA_Ij*0bGJtWl?
N_fwK`!d^>ps7&g-SU9ajYBuGth3vMgYK^Xpw7qbkqR&3p!Zh1i*Q5(@e8DICY3UeAY}zuvr8WM380YA>D34W=3s$Ko~M=JzpeBT
cV0pLBRk#_@rFTH>!<~_O=8}$+x@`5b^7{NiU!*h+(Tp`cUIFlrR+I)O{VETGR?Qx!%29Y`rTyG+hg&ALQut}VMv?;s2nzGJ_FX0
T5kK8$#DJyx7B22lrEl##^$Atwi4v0i23y&3(w8C;Qo|e6JNIi>9fiJ8tGclZt=B8TkczfsC_Iz-#IsO`89IMJ2F7yFb!+P@OM_x
e>bF8oo8G4;^=1vtwtD#dfM!Z(vloM&NNI@yR5xAv$bw5S*W3iLc%XU4d9-Z=Q31}W~@tvQI;V$x2U@ORc-~53h0vw*R6CvE~78M
fuJ-m1hZ5>EI&Ho2N|lot7%<)Qx;Wx&AM}P!IQwM-BX+goFHPtyvQ9)A;REeIG#`ePI~=>Utjy#&Ck;_s{$oK3$osuaBu20YVn8Q
nBnVz2#xl}0S_;4{xn4U(ganiz{#uPKvXF1-R8Z4u{X;O>~L?2^g?yI{D}}PgX9DGb=(u2@b}6G;WA!+eaMuEHX0>lfd-dp#ENj%
b*zA%pijG%Fj?j|d?5P(dJuQ<cJkI`1q$fq0m=Pyc*4$$L9tMZ+=91+pZeYTI08g_Bm)+B1h0~zNW%1@Rvksq7jzJudW(L7!Wp!e
hx~5A3JDdPy;Zl+3Eq$s1)XV-NK6tbr$=ytezn=7vtBx#iKBXswn*h!5kasC56Vf;iS^+o4I?#;@%yilN2g8XPAWUyozbmdhkh=1
kWR}9_`lVdDs<=bQC8Szj~`>~C=ltNiV3xlp20cqEp*wd@I7W?{nNbT>-rFkM2TUOrM~x<_Job7!car7-|m!);6GS%8G6!cKSp#z
Lx^dNx@g+P0W0r@&Z91T%jS|q+apkT7<(7tD4%@2uicJ%*kH%Z<~?IH-DZYY9z;qForN*szylH0G<cv=vHQKXNrS7lk-1f5j$2r<
mGDZT<B`HMXqv2zE6zPKiu!2V1b2IU%6e*}M@K`5D-|JvHy`+`BEQ7CFlo62WCHg_gS+jY(pkWTZMy)gSe;gGEi%2J=;^E?m?DrA
neZF{#(P=FB-ZCC00EGw0l0(+c?+SuvBYQl0ssI200dcD
"""


def _rebuild(p):
    out = []
    for ti, ci, pr, am, lm, nt, nb in p["rows"]:
        t = p["traces"][ti]
        tv = dict(zip(p["trace_keys"], t[2:]))
        d = {}
        for k in p["order"]:
            if k == "question_id": d[k] = t[0]
            elif k == "hint": d[k] = t[1]
            elif k == "condition": d[k] = p["conds"][ci]
            elif k == "probs": d[k] = dict(zip("ABCD", pr))
            elif k == "argmax": d[k] = am
            elif k == "letter_mass": d[k] = lm
            elif k == "n_input_tokens": d[k] = nt
            elif k == "n_blocked": d[k] = nb
            elif k in p["const"]: d[k] = p["const"][k]
            else: d[k] = tv[k]
        out.append(json.dumps(d, ensure_ascii=False) + "\n")
    return "".join(out)


def restore_tier1():
    existing = [f for f in RUN_DIR.glob("*.jsonl")
                if f.stat().st_size and not f.name.endswith((".selection.jsonl", ".errors.jsonl"))]
    if existing:
        print(f"results already in {RUN_DIR}; tier-1 data cell skipped")
        return
    pack = json.loads(lzma.decompress(base64.b85decode("".join(BLOB.split()))))
    files = dict(pack["files"])
    files["olmo-3-7b-think.jsonl"] = _rebuild(pack["records"])
    for name, text in files.items():
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if got != SHA256[name]:
            raise SystemExit(f"tier-1 data: checksum mismatch for {name}; nothing written")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (RUN_DIR / name).write_bytes(text.encode("utf-8"))
    n = files["olmo-3-7b-think.jsonl"].count("\n")
    print(f"tier-1 results restored into {RUN_DIR}: {n} records, all checksums OK")


restore_tier1()
