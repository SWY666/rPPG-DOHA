from copy import deepcopy
import numpy as np
import torch

class GHM_module:
    # 让网络不过分关注于困难的样本。
    def __init__(self, len_dataset):
        self.pool = [-1 for x in range(len_dataset)]
        self.pool = [0.9004666209220886, 0.8922725915908813, 1.1302090883255005, 0.8251768350601196, 0.8055906891822815, 0.7301729917526245, 0.6997647285461426, 0.6946346163749695, 0.8485568761825562, 1.2414811849594116, 0.7053670883178711, 0.8229874968528748, 1.083901286125183, 0.973705530166626, 0.9096897840499878, 0.6478174328804016, 0.8117641806602478, 0.643324613571167, 0.6771218776702881, 0.8004364967346191, 0.5529803037643433, 0.6534397602081299, 0.8481499552726746, 0.7147968411445618, 0.7200198769569397, 0.5807443261146545, 0.6811138391494751, 0.545137345790863, 0.47244128584861755, 1.2473151683807373, 0.7791691422462463, 0.887604296207428, 0.7110562324523926, 0.623816192150116, 1.0392252206802368, 0.8514143228530884, 0.9310247898101807, 1.257917046546936, 1.164683222770691, 0.5877495408058167, 0.6837285161018372, 0.929955780506134, 0.7794855833053589, 0.7761445045471191, 0.9713777303695679, 0.8481619358062744, 0.6020439267158508, 0.9210599064826965, 0.8488905429840088, 0.9342349767684937, 0.9406587481498718, 0.7019638419151306, 0.9877917170524597, 1.015957236289978, 0.6627371311187744, 0.718757688999176, 0.8513728380203247, 0.758094847202301, 1.0436267852783203, 0.6960560083389282, 0.9918158054351807, 0.7291374802589417, 0.7919952273368835, 0.7285076975822449, 0.6279601454734802, 0.8540675044059753, 0.8590189218521118, 1.0642439126968384, 0.9137718677520752, 0.80118727684021, 0.8092401027679443, 0.908686637878418, 0.872654914855957, 0.7897526621818542, 0.8471550345420837, 1.1913976669311523, 1.1557811498641968, 0.8174482583999634, 1.0039222240447998, 0.6206245422363281, 0.5453665256500244, 1.0193299055099487, 0.7985180616378784, 0.7141819596290588, 1.2047141790390015, 0.7452230453491211, 0.44948068261146545, 1.001043438911438, 0.6786914467811584, 0.6113560199737549, 0.20453698933124542, 0.575193464756012, 0.9839727878570557, 0.5733464956283569, 1.03648042678833, 0.9972703456878662, 0.7725076079368591, 0.7788212895393372, 0.6701899766921997, 0.8650211095809937, 0.8912355303764343, 0.7459348440170288, 0.8175408840179443, 0.2618001103401184, 0.7755364775657654, 0.8380599617958069, 1.0551801919937134, 1.2815428972244263, 0.7960206866264343, 0.6322904229164124, 0.9113003015518188, 0.8203694224357605, 0.7860238552093506, 0.8083178400993347, 0.630993664264679, 0.8394741415977478, 0.7399381399154663, 0.6767064332962036, 0.7815855145454407, 0.569506824016571, 0.7742848992347717, 0.8754621148109436, 1.077162504196167, 1.316489815711975, 0.7404699921607971, 0.7796647548675537, 0.9110921025276184, 0.8154087662696838, 0.6986212134361267, 0.7879464626312256, 0.6792868375778198, 0.6363884210586548, 1.0983054637908936, 0.8288799524307251, 1.0925006866455078, 0.9521828293800354, 0.5213596224784851, 1.1003239154815674, 0.8213537931442261, 0.7653047442436218, 1.111385464668274, 0.6048583984375, 0.5333108305931091, 0.822144091129303, 0.8424655199050903, 0.7833266258239746, 0.6248170733451843, 0.8609393835067749, 1.141575574874878, 0.8957657217979431, 0.6411891579627991, 1.232802152633667, 1.049521565437317, 0.7677632570266724, 0.7871143221855164, 0.6199175119400024, 0.7591412663459778, 0.9489437341690063, 0.9761185646057129, 0.4903428256511688, 0.8934061527252197, 0.8954607248306274, 0.9007803201675415, 1.1695529222488403, 0.9320949912071228, 1.4517008066177368, 0.5587689876556396, 0.927044689655304, 0.8918578624725342, 0.742150604724884, 0.6552824378013611, 1.331907033920288, 1.1081314086914062, 1.12542724609375, 0.9611997604370117, 0.8798711895942688, 0.714034378528595, 1.0538420677185059, 0.6069740653038025, 0.9205721616744995, 1.0031441450119019, 0.7484932541847229, 0.7885276675224304, 1.0890789031982422, 0.7811524868011475, 0.6925087571144104, 0.6401081085205078, 0.6605948209762573, 0.7102998495101929, 0.9302595257759094, 0.4616553783416748, 0.5957198143005371, 0.9013031125068665, 0.6786909103393555, 1.135212779045105, 0.837984025478363, 0.9303962588310242, 0.7662449479103088, 0.6793386936187744, 1.1857991218566895, 0.7390227317810059, 1.1782422065734863, 0.9690111875534058, 1.0834999084472656, 0.7993091344833374, 0.8591973781585693, 0.8676474094390869, 0.9811010956764221, 0.5807727575302124, 0.841979444026947, 0.7485954761505127, 0.6782059073448181, 0.9637834429740906, 0.7437389492988586, 0.8850060105323792, 1.0249240398406982, 0.8988752365112305, 1.058133602142334, 0.6534637212753296, 1.012678861618042, 0.7335875630378723, 0.7749532461166382, 0.5704832077026367, 1.0791983604431152, 0.5390846729278564, 0.8752242922782898, 0.7137737274169922, 0.7476071119308472, 0.816520094871521, 0.8159813284873962, 0.7244275212287903, 0.6860834360122681, 0.7412804365158081, 0.6521603465080261, 0.89235919713974, 0.9150348901748657, 0.8864517211914062, 1.057875156402588, 1.01531183719635, 0.7333678603172302, 0.554840624332428, 0.8261585831642151, 0.7953008413314819, 0.8321199417114258, 0.9364206194877625, 1.1097378730773926, 0.4969862997531891, 0.7897422313690186, 0.821662187576294, 0.731667697429657, 0.7067924737930298, 1.1538950204849243, 0.6624115109443665, 0.7798842191696167, 0.6388686299324036, 0.6307232975959778, 0.5876750349998474, 1.0270975828170776, 0.7066418528556824, 0.6701754927635193, 0.7953596115112305, 1.1262015104293823, 0.8094022870063782, 0.7458590865135193, 1.0337151288986206, 0.7516486644744873, 0.8567238450050354, 0.4446505010128021, 0.6885288953781128, 1.1633118391036987, 0.6519261598587036, 0.8125078082084656, 0.8699530363082886, 0.6479381918907166, 0.8188794851303101, 0.7331987619400024, 1.0241600275039673, 0.7637912631034851, 1.0224496126174927, 1.2176321744918823, 0.9811354279518127, 0.6789606809616089, 0.8313442468643188, 0.753690242767334, 0.7437616586685181, 0.7805635929107666, 0.9238275289535522, 0.7236318588256836, 1.0632561445236206, 0.22073331475257874, 1.159386396408081, 0.9935793876647949, 0.9406527280807495, 0.7922879457473755, 0.7640426158905029, 0.6831265091896057, 1.0303857326507568, 0.7024503350257874, 0.8434582948684692, 0.8850056529045105, 0.8511644601821899, 0.759407103061676, 0.8470368385314941, 0.6030279994010925, 0.9679805040359497, 0.7764760255813599, 0.7806910872459412, 0.8330150842666626, 0.8457052707672119, 0.8411116600036621, 0.9651390910148621, 0.7124203443527222, 0.6183397769927979, 0.6448563933372498, 1.3549144268035889, 1.1902356147766113, 1.4249961376190186, 1.1282438039779663, 0.6976519823074341, 0.7995943427085876, 1.1345598697662354, 0.7571934461593628, 0.9042292833328247, 0.8205702900886536, 0.9425037503242493, 0.8241773843765259, 1.1428487300872803, 1.0240932703018188, 0.7407487630844116, 0.8999212384223938, 0.7928580045700073, 0.3355119228363037, 0.7593963742256165, 0.9535488486289978, 1.123714566230774, 0.881767988204956, 0.5963064432144165, 0.9201275110244751, 0.9948599934577942, 0.8029512166976929, 0.7697223424911499, 0.9207861423492432, 0.8914182782173157, 0.7216550707817078, 0.8452014327049255, 0.6911696195602417, 0.8350418210029602, 0.8137699365615845, 0.831850528717041, 0.958071231842041, 1.0910983085632324, 0.6890605092048645, 0.9494677782058716, 0.8394569754600525, 0.9664871692657471, 1.089605450630188, 0.7488893866539001, 0.7009528279304504, 0.8665304183959961, 0.7481233477592468, 0.9544599652290344, 0.9040433168411255, 1.0402638912200928, 0.8437458872795105, 0.7070677280426025, 0.7583805918693542, 0.7041252255439758, 1.2160124778747559, 1.0542114973068237, 1.1041921377182007, 1.3031550645828247, 0.7694545984268188, 1.1227900981903076, 0.884669303894043, 1.2681102752685547, 0.944858968257904, 0.8383429050445557, 0.5945310592651367, 0.964244544506073, 0.8548228144645691, 0.8400624990463257, 0.8186854124069214, 0.9908876419067383, 0.8409225344657898, 0.642623782157898, 0.6659771800041199, 0.70609450340271, 0.517700731754303, 0.986301064491272, 0.7165794968605042, 0.9287058115005493, 0.6017299294471741, 0.9154587388038635, 0.7275142073631287, 1.261083722114563, 0.9448518753051758, 0.9630407094955444, 1.019319772720337, 0.6606062054634094, 0.8248956203460693, 0.7153810858726501, 0.7572830319404602, 1.0773924589157104, 0.6349958777427673, 1.173602819442749, 0.8282892107963562, 0.7987899780273438, 0.5025854110717773, 0.4705750346183777, 0.7891560792922974, 0.8244443535804749, 0.5851118564605713, 0.9061108827590942, 0.8866259455680847, 0.7592188715934753, 1.0343878269195557, 0.8336806893348694, 0.6911513805389404, 0.894200325012207, 0.6325609087944031, 0.8199943900108337, 1.1131749153137207, 0.7748073935508728, 1.0896798372268677, 0.8096112012863159, 0.6445763111114502, 0.9180853366851807, 0.9052681922912598, 0.9548983573913574, 1.5982789993286133, 0.9958118796348572, 0.8937156200408936, 1.3106309175491333, 0.5745081305503845, 0.9337396025657654, 0.7858127355575562, 0.8260036110877991, 0.9616552591323853, 0.7442670464515686, 0.7737320065498352, 1.2029445171356201, 0.8165382742881775, 0.8681880235671997, 0.7905905246734619, 0.6371889114379883, 0.5459747314453125, 1.2273319959640503, 1.126508116722107, 0.8989399671554565, 0.7633101940155029, 0.97805255651474, 0.6208183169364929, 0.540187418460846, 0.5735976696014404, 0.7376907467842102, 1.2121001482009888, 0.8325720429420471, 0.9431498050689697, 0.8465799689292908, 0.8989946842193604, 0.8906940221786499, 0.9045299291610718, 0.8065567016601562, 0.8065171241760254, 0.6474101543426514, 0.6044906377792358, 0.7639591097831726, 0.6515678763389587, 0.8813385963439941, 1.1229993104934692, 0.8238258957862854, 0.6656872034072876, 0.937468945980072, 0.9557592868804932, 0.7414878606796265, 0.7222182154655457, 1.0117144584655762, 0.8129194378852844, 0.8432024717330933, 0.5080240964889526, 0.564533531665802, 0.5601394772529602, 1.6286605596542358, 0.6514736413955688, 1.2928438186645508, 0.7816445827484131, 0.7365714311599731, 0.7085297107696533, 0.7594220042228699, 0.9384068250656128, 0.8935322165489197, 0.6355831027030945, 0.729005753993988, 0.8884751200675964, 0.9263491034507751, 0.4332800507545471, 0.6024021506309509, 0.6826278567314148, 0.6725316643714905, 0.612855851650238, 0.8222967386245728, 0.7128232717514038, 0.9232850074768066, 0.60828697681427, 0.7239046096801758, 0.5809475779533386, 0.9498029947280884, 1.0465292930603027, 0.5523343682289124, 0.9237787127494812, 0.5931485891342163, 0.5155089497566223, 0.6270632743835449, 0.5937833189964294, 0.7485954165458679, 0.7742723226547241, 0.9804691672325134, 0.8336681723594666, 0.7919683456420898, 0.6836516261100769, 0.9292072057723999, 0.8505827188491821, 0.6531599760055542, 1.1991751194000244, 0.8718912601470947, 0.9948922991752625, 0.8707022666931152, 0.1860887110233307, 0.853365957736969, 1.007650375366211, 0.8770737648010254, 0.7958459258079529, 0.8017914295196533, 0.7141464352607727, 0.9231352806091309, 0.5553173422813416, 0.756475567817688, 1.1236903667449951, 0.9038227200508118, 0.9819312691688538, 0.7305457592010498, 0.7524060010910034, 0.6981694102287292, 0.8940972685813904, 0.8792349100112915, 0.6701185703277588, 1.1744861602783203, 0.790217936038971, 1.1830424070358276, 0.8459910750389099, 0.8642638325691223, 0.7024970650672913, 0.7693750262260437, 0.8102725744247437, 1.0318082571029663, 0.7224436402320862, 0.9633966684341431, 0.5958776473999023, 0.8407856822013855, 0.9545902013778687, 1.1230504512786865, 0.8717976212501526, 0.9091284871101379, 1.0093377828598022, 0.5974621176719666, 0.8216968178749084, 0.7190942168235779, 0.7247477173805237, 0.820666491985321, 1.0349667072296143, 0.988165020942688, 0.39654675126075745, 0.7694732546806335, 0.9838747978210449, 0.9597166180610657, 0.8163381814956665, 0.9131237864494324, 0.9602904319763184, 0.9951967000961304, 0.7316417098045349, 0.8786774277687073, 0.9659711122512817, 1.1982430219650269, 0.8859076499938965, 0.6947948336601257, 0.7130468487739563, 1.2523590326309204, 0.9395503401756287, 0.7374112010002136, 0.6462764143943787, 0.8076545596122742, 0.7083985209465027, 0.6503450870513916, 1.2845051288604736, 1.2517006397247314, 1.0571850538253784, 0.9294885396957397, 0.8467169404029846, 0.809005618095398, 0.8124192953109741, 0.6504348516464233, 0.6481034755706787, 0.8044307827949524, 0.6432563662528992, 0.6198182702064514, 0.5135397911071777, 0.9159107804298401, 1.1291640996932983, 0.7468082308769226, 1.1380609273910522, 0.7001346945762634, 0.7249945998191833, 0.7754120826721191, 0.8902315497398376, 1.1824272871017456, 0.912605345249176, 0.7557111978530884, 1.2372785806655884, 0.7700133323669434, 0.7851644158363342, 1.098432183265686, 0.7924417853355408, 0.9112169742584229, 0.7233672738075256, 1.04619300365448, 0.8923792243003845, 0.6139059662818909, 0.5929033160209656, 0.8095272183418274, 0.8885859847068787, 1.2257763147354126, 0.7857806086540222, 0.9042115211486816, 1.1827068328857422, 0.6119205355644226, 0.9295557737350464, 0.5862995386123657, 1.0642441511154175, 1.0627373456954956, 0.806279182434082, 0.9232192039489746, 0.952637791633606, 1.0600712299346924, 0.5646094083786011, 0.9837397933006287, 0.9204421043395996, 0.7892290353775024, 0.5804737210273743, 0.6026118993759155, 0.6681340336799622, 0.871856153011322, 0.9266763925552368, 0.7286857962608337, 0.7223458886146545, 0.6780490279197693, 0.8642287850379944, 0.4724167287349701, 0.9807612299919128, 1.153242826461792, 0.8256701231002808, 0.9622150659561157, 0.9967195987701416, 1.1826212406158447, 0.6695795059204102, 1.1103785037994385, 0.9995706081390381, 0.7750238180160522, 0.7536175847053528, 1.0515079498291016, 1.1290894746780396, 0.6632715463638306, 0.9960076212882996, 0.7921480536460876, 0.5963540077209473, 0.995843231678009, 0.8842125535011292, 0.8618263602256775, 0.9475655555725098, 0.5684706568717957, 0.7607371211051941, 0.6105692386627197, 0.9091005325317383, 0.7514005899429321, 1.020331621170044, 1.4389442205429077, 1.0425879955291748, 1.0152298212051392, 0.5234091281890869, 0.5843369364738464, 0.7385008335113525, 0.7423606514930725, 0.805797815322876, 1.0183343887329102, 0.749027669429779, 0.8959531784057617, 1.053144931793213, 0.8299451470375061, 0.8380791544914246, 1.045775055885315, 0.7352191209793091, 0.6714618802070618, 0.7094759345054626, 1.2158902883529663, 0.649431049823761, 1.0796724557876587, 0.9156734347343445, 0.7124170064926147, 0.825010359287262, 0.7451196908950806, 0.5065684914588928, 0.6532743573188782, 0.935761570930481, 0.9101400971412659, 0.6409256458282471, 0.7820571064949036, 0.9007128477096558, 0.7877829074859619, 0.5410997867584229, 0.7308433651924133, 0.8194454312324524, 0.5984615683555603, 0.7782594561576843, 0.8104689717292786, 0.9181734919548035, 0.9010376930236816, 0.6416073441505432, 0.6711776852607727, 0.7406970262527466, 0.7682161331176758, 0.997564435005188]
        self.bin = 10
        self.bar = int(len_dataset * 0.07)
        self.max_num = len_dataset
        self.wane_alpha = 0.5
        self.handle = True

    def push(self, reg_loss, index):
        batch_size = reg_loss.shape[0]
        index_list = [index[idx].item() for idx in range(batch_size)]
        for idx in range(batch_size):
            self.pool[int(index_list[idx])] = reg_loss[idx].item()

        if self.check_unifromity():
            return self.return_weight(reg_loss).detach()
        else:
            result_item = torch.ones([reg_loss.shape[0]]).detach()
            return result_item

    # check the uniformity of the self.pool
    def check_unifromity(self):
        if -1 in self.pool:
            return False
        else:
            if self.handle:
                print(self.pool)
                self.handle = False
            return True

    def return_weight(self, reg_loss):
        reference_list = deepcopy(self.pool)
        reference_list.sort()
        result_item = torch.ones([reg_loss.shape[0]])
        for idx in range(reg_loss.shape[0]):
            num_id = reference_list.index(reg_loss[idx].item())
            if num_id <= self.bar:
                # print("小", num_id - self.bar, (0.3 + 0.7 * np.exp((num_id - self.bar)/(self.max_num / 20))))
                result_item[idx] *= (0.3 + 0.7 * np.exp((num_id - self.bar)/(self.max_num / 20)))
            if num_id >= self.max_num - self.bar:
                # print("大", num_id - self.bar, np.exp((self.max_num - self.bar - num_id)/(self.max_num / 20)))
                result_item[idx] *= (0.3 + 0.7 * np.exp((self.max_num - self.bar - num_id)/(self.max_num / 20)))

        return result_item
