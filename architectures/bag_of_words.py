from base64 import encode
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm

def bag_of_words(x, y):
    # class_one = set(["government","military","war","army","troops","soldiers","forces","civilian","soldier","combat","militia","wars","conflict","force","officers","commanders","battle","troop","pentagon","wartime","armed","peace","warfare","hostilities","offensive","generals","command","commander","ceasefire","leadership","civilians","occupation","governments","country","violence","warlord","officials","invasion","authorities","terroristsist","general","political","defense","rebel","rebels","insurgents","deployment","fighters","aggression","fighting","victory","involvement","crisis","regime","enemy","peaceful","terrorism","marines","led","mission","attack","guerrillas","politicians","violent","weapons","insurgency","nation","allied","stationed","truce","massive","govt","personnel","ministry","units","state","critical","defence","leader","reconstruction","paramilitary","militias","presence","resistance","navy","headquarters","campaign","leaders","terrorist","tensions","strategic","intelligence","action","coup","official","enemies","material","militants","officer","rocket","mercenary","fire","major","allies","administrationkeepers","operational","coalition","foreign","undermine","servicemen","invaded","operation","conflicts","guard","president","the","assault","battlefield","national","base","cabinet","operations","defeat","mercenaries","emergency","situation","outpost","states","firing","sabotage","terror","program","tanks","opponentsc","bombing","intervention","stronghold","agents","politician","division","domestic","service","effective","ruler","rebellion","security","attacks","occupied","agencies","deployed","stabilize","american","battles","strikes","public","send","naval","governing","territory","police","power","powers","senior","pull","ruling","serving","captured","system","politics","assaults","office","warrior","tension","hostile","diplomat","blockade","minister","agency","rulers","draft","installations","rule","region","guerrilla","unitries","federal","reinforce","opposition","city","uprising","disasteringon","militant","tank","embassy","fortified","hq","premier","dictatorship","tactical","turmoil","gun","artillery","deadly","s","siege","heavy","practice","patrioticarian","sovereignty","missiles","programme","zones","sector","ally","covert","policy","bloody","chaotic","killed","missile","countries","move","british","threat","for","abandon","americans","dictator","defenses","anarchyvocation","positions","onslaught","politically","harsh","threats","taliban","bureaucracy","strike","withdrawal","fresh","party","peacekeeping","strategyurgent","bombardment","crowded","propaganda","withdraw","systems","unrest","movement","deposed","staff","moves","specialists","stability","group"])
    # class_two = set(["basketball","football","athletes","sports","athletic","athlete","sporting","players","coaches","player","baseball","nba","sport","nfl","hockey","quarterback","game","soccer","play","team","games","coach","footballer","playing","athletics","bowl","footballers","league","lineman","teams","knicks","ncaa","offensive","tournament","defenders","sprinter","season","plays","gymnastics","championships","played","rugby","match","receiver","professional","cyclist","softball","coaching","seasons","stadium","volleyball","arena","mlb","forward","championship","playoff","champion","tigers","teammates","defender","track","linebacker","club","goal","kicker","offense","cup","elite","running","nhl","wrestling","nascar","clubs","college","boxing","handballrs","center","golf","compete","racing","scorer","coached","backs","fa","striker","franchises","pitcher","postseason","tackle","lions","star","scoring","contests","talents","afc","falcons","offenses","cricket","eagles","olympic","medal","fans","swimmersathlon","program","preseason","pro","winning","arenas","practice","highlight","qb","turf","exhibitioners","defensive","pistonsf","squadsbuilding","fifa","national","coltspers","events","rookie","showdown","bowls","state","wide","juventus","tennis","ball","guard","kick","boxer","field","celtics","win","mvp","the","matchess","stadiums","mavericks","unbeaten","champions","competition","rowing","broadcasting","slam","playoffs","side","touchdown","midfielder","series","polo","goalkeeper","lakers","cycling","raiders","cornerback","olympics","patriots","squad","court","recreational","franchise","gymnast","head","shooting","boxers","redskins","lacrosse","bowler","dolphins","defense","stars","coliseum","locker","cavaliers","cardinals","doping","goaltender","espn","clips","cyclists","training","batting","rushing","f1","tournaments","competitive","fight","receivers","racers","rower","internationals","passing","sprint","hitter","sox","rebound","fights","49ers","cfl","winner","race","finals","teammate","fullback","rivalry","ends","bulldogs","premiership","talent","campaign","starters","basket","trainer","mariners","groundcs","swimmer","uniform","career","baskets","huskiesback","wrestlerser","bucks","drivers","strikersinggueball","floor","rb","titans","major","music","big","outfielder","military","cricketer","quarterfinal","medalist","savior","performances","association","against","semifinals","buccaneersu","guards","comeback","motorsport","indoor","run","swim","place","kingsline","chargers"])
    # class_three = set(["stocks","markets","industries","companies","market","businesses","earnings","firms","prices","futures","traders","shares","sectors","investors","trades","sales","contracts","merchants","assets","options","institutions","stock","industry","exports","reserves","trading","customers","commodities","brandsties","trade","profits","manufacturers","economic","manufacturing","shipments","securities","holdings","investments","resources","producers","offerings","trader","products","funds","marketplace","goods","operationsmarksies","losses","revenues","investment","yields","bonds","purchases","stores","financial","corporations","company","retailers","selling","business","values","income","buying","corporate","index","retail","orders","consumer","competition","indices","economy","rangesmakers","practices","exchanges","banks","consumerss","enterprises","reports","sector","deals","pricing","partnersers","suppliespile","traded","gains","shareholders","growth","factoriesulator","positions","materials","unitskeres","production","economiesex","expectations","employers","machinery","buyers","rates","inflation","entrepreneurs","demand","installations","output","indicatorsbbies","subsidies","bankers","returns","issues","producing","firm","flows","inventory","buyer","interests","acquisitions","executives","regulators","500","plants","targets","managersoffs","advisers","packagestions","imports","participants","sell","sentiment","dollar","oil","slips","priceics","industrial","alliances","giant","priced","shops","services","moves","owners","vendors","revenue","clients","dealer","transactions","entities","dealers","loansbilities","makers","popular","competitors","advertisingries","boards","businessmen","estimates","commercial","lines","advertisements","budgets","sale","properties","target","buysncies","items","deliveries","metals","showing","platforms","streetps","costs","competitive","policies","investor","relations","debts","value","cash","exportn","accounts","suppliers","claims","fund","outfits","holders","share","trends","listing","terminals","fields","players","casinos","segments","projections","spending","partnershipstrust","commissions","store","fuels","buy","facilitiesoa","instruments","quotes","volumes","dealings","listings","increases","componentsoutsbacks","opportunities","marketing","producer","analysts","barrels","operators","wealth","profitmarket","offering","groups","sold","chemicals","activityeller","slides","creditors","peers","concessions","giants","chains","commodity","asset","utilityg","financesil","boxes","farmers","confidence","consumptioncies","mergers","purchase","displays"])
    # class_four = set(["computer","telescope","software","computers","computing","hardware","technology","system","pcer","machine","it","systemsware","applications","program","chip","programs","server","desktop","electronic","code","simulator","electronics","machines","database","technologies","internet","chips","infrastructure","devices","browser","device","processor","tools","phone","specification","application","dataulator","utility","processors","laptop","architecture","spacecraft","linux","networking","platformdget","servers","superl","pcs","tool","programming","microsoft","patch","engine","graphicstor","components","equipment","instrument","appicom","cpu","micro","programmer","tech","drive","digitalwall","networks","servicersorers","cards","platformsdgets","notebook","protocol","memory","semiconductorp","satellite","keyboard","operating","8","packageboard","company","products","languagetation","displaysrator","lan","processingm","features","rocket","robot","development","security","exploit","satellitesxjetedot","player","enterprise","game","receivers","display","standard","com","networkbook","probe","controller","windows","instruments","installations","performance","mackis","clusters","product","wireless","video","multimediacecore","robots","telescopes","sensors","project","switch","source","observatory","dynamic","solution","embeddedte","phones","cyber","suite","specialized","screen","solutions","mechanisms","communication","technical","popular","center","unix","framework","os","user","vehicle","astronomy","games","telecommunications","online","client","laserier","console","card","printers","informationetseron","sensor","version","massivena","downloadia","implanting","info","interceptor","peripheral","players","array","monitor","services","pioneer","technological","vulnerabilityam","update","designrinoum","macintoshnet","box","proprietaryrs","mouse","boardsization","batteries","upgrade","booster","web","developers","feature","content","unitu","radioapp","media","2","nasatech","panel","mechanismbot","ibm","filetime","install","blade","panels","analog","spaceshipdps","core","missilemax","spaceatics","stackamp","driveslla","wi","programmerskit","automated","pro","freeicsac","patches","component","communications","portet","optical","intel","flight","buffer","clusterky","researchice"])
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    class_one = ["government military war army troops soldiers forces civilian soldier combat militia wars conflict force officers commanders battle troop pentagon wartime armed peace warfare hostilities offensive generals command commander ceasefire leadership civilians occupation governments country violence warlord officials invasion authorities terroristsist general political defense rebel rebels insurgents deployment fighters aggression fighting victory involvement crisis regime enemy peaceful terrorism marines led mission attack guerrillas politicians violent weapons insurgency nation allied stationed truce massive govt personnel ministry units state critical defence leader reconstruction paramilitary militias presence resistance navy headquarters campaign leaders terrorist tensions strategic intelligence action coup official enemies material militants officer rocket mercenary fire major allies administrationkeepers operational coalition foreign undermine servicemen invaded operation conflicts guard president the assault battlefield national base cabinet operations defeat mercenaries emergency situation outpost states firing sabotage terror program tanks opponentsc bombing intervention stronghold agents politician division domestic service effective ruler rebellion security attacks occupied agencies deployed stabilize american battles strikes public send naval governing territory police power powers senior pull ruling serving captured system politics assaults office warrior tension hostile diplomat blockade minister agency rulers draft installations rule region guerrilla unitries federal reinforce opposition city uprising disasteringon militant tank embassy fortified hq premier dictatorship tactical turmoil gun artillery deadly s siege heavy practice patrioticarian sovereignty missiles programme zones sector ally covert policy bloody chaotic killed missile countries move british threat for abandon americans dictator defenses anarchyvocation positions onslaught politically harsh threats taliban bureaucracy strike withdrawal fresh party peacekeeping strategyurgent bombardment crowded propaganda withdraw systems unrest movement deposed staff moves specialists stability group"]
    class_one = ["military government syria invasion humanitarian genocide war lebanon conflict occupation afghanistan interference militias movement diplomatic syrian coalition peacekeeping allies resistance ceasefire insurgency kidnappers regime corruption stability spiraling delegation hunger torture taliban kuwait aides occupied affairs diplomat legislative sudan bloodshed protests chechnya army sudanese imposed arms troop fallujah terrorism terrorists militia damascus aide peninsula factions fighters tribunal beirut commander response crimes restive kabul immigration prominent terrorist lebanese ambassador expert envoy straw separatists diplomats observers soldiers rebels unity widen darfur parties martial ruler yawar reconstruction leadership detention armitage territories bail covert prosecutor closure atrocities mideast falluja guerrillas terror ivorian calm saddam duty counter interests burma obasanjo uprising fragile courts crisis stronghold violence samarra kidnapping loyalists barrier temporary region khartoum ethnic commanders muslims patrol tightened chaos upsurge faction questioning envoys hussein disarm afghan trap organizing massacre appealed pentagon un resignations witness tensions constitutional surrender detainees israel stable negotiator redeploy demands involvement fence flared provincial govt gbagbo politician herat forces disrupt jihad ministry compound misconduct convoy nato parliamentary trafficking ally dangers redeployment deadlocked politicians extradition disputed pledge gloria fate opposed shalom abroad basque mounted refugee elections intervention allawi mosul disarmament armored tribal interior beheading extremist families urging embassies undermine blockade takers pledging kidnappings chilean cloning delegates arabia ignoring videotape arroyo tripoli jails lavrov divided hague islamist"]
    class_two = ["basketball football athletes sports athletic athlete sporting players coaches player baseball nba sport nfl hockey quarterback game soccer play team games coach footballer playing athletics bowl footballers league lineman teams knicks ncaa offensive tournament defenders sprinter season plays gymnastics championships played rugby match receiver professional cyclist softball coaching seasons stadium volleyball arena mlb forward championship playoff champion tigers teammates defender track linebacker club goal kicker offense cup elite running nhl wrestling nascar clubs college boxing handballrs center golf compete racing scorer coached backs fa striker franchises pitcher postseason tackle lions star scoring contests talents afc falcons offenses cricket eagles olympic medal fans swimmersathlon program preseason pro winning arenas practice highlight qb turf exhibitioners defensive pistonsf squadsbuilding fifa national coltspers events rookie showdown bowls state wide juventus tennis ball guard kick boxer field celtics win mvp the matchess stadiums mavericks unbeaten champions competition rowing broadcasting slam playoffs side touchdown midfielder series polo goalkeeper lakers cycling raiders cornerback olympics patriots squad court recreational franchise gymnast head shooting boxers redskins lacrosse bowler dolphins defense stars coliseum locker cavaliers cardinals doping goaltender espn clips cyclists training batting rushing f1 tournaments competitive fight receivers racers rower internationals passing sprint hitter sox rebound fights 49ers cfl winner race finals teammate fullback rivalry ends bulldogs premiership talent campaign starters basket trainer mariners groundcs swimmer uniform career baskets huskiesback wrestlerser bucks drivers strikersinggueball floor rb titans major music big outfielder military cricketer quarterfinal medalist savior performances association against semifinals buccaneersu guards comeback motorsport indoor run swim place kingsline chargers"]
    class_two = ["basketball coaches football coaching college elite junior valuable cycling athletes rookies swimming goalkeeper boxing talented undefeated dreams bode scottish fame performances marquee offseason prolific prestigious sidelines honor weaver bcs nebraska contenders hockey quarterbacks franchise walter judging wade pennant relay vacant ultimate greene 35th mens storied soccer triathlon 1918 miller youngest mates crowned richest trojans tilt titles midfield greatest legend fifa matchup rico alpine famous gymnastics uniform hopkins greet sprinter reminded squad consideration birthday greeks venue coordinator wnba recognized matthew teammate bowled toughest sounded bernhard contest races olympian memories softball golfer alumni burglary rugby televised crushed leagues weightlifting jamaican blues sole wrestling tradition klinsmann charlton nightmare bat mickelson roster portuguese volleyball ut felix bronze weekends alabama badgers favourites urban perfection felt mls circle vaughan hearts martnez shortstop swimmers regarded tarnished magical spectacular podium finalists payton wore honors glory homestead willingham knockout boys oswalt rowing tenth standings silver played artistic precious secondary cats valentino rated staying mardy monty english middleweight pride spammed dodger arkansas zook coached gary hometown favorites undisputed sitting seniors rotation destined silent blackburn joy cavaliers montgomerie benched heck phil nationals medley annika puerto virtually megson wakefield singled toss truex howell croom monza legendary friend celebrating han vacancy awful accomplished locked shadows magnificent raucous eligible rallying 16th carly notice spartans keith clubs"]
    class_three = ["stocks markets industries companies market businesses earnings firms prices futures traders shares sectors investors trades sales contracts merchants assets options institutions stock industry exports reserves trading customers commodities brandsties trade profits manufacturers economic manufacturing shipments securities holdings investments resources producers offerings trader products funds marketplace goods operationsmarksies losses revenues investment yields bonds purchases stores financial corporations company retailers selling business values income buying corporate index retail orders consumer competition indices economy rangesmakers practices exchanges banks consumerss enterprises reports sector deals pricing partnersers suppliespile traded gains shareholders growth factoriesulator positions materials unitskeres production economiesex expectations employers machinery buyers rates inflation entrepreneurs demand installations output indicatorsbbies subsidies bankers returns issues producing firm flows inventory buyer interests acquisitions executives regulators 500 plants targets managersoffs advisers packagestions imports participants sell sentiment dollar oil slips priceics industrial alliances giant priced shops services moves owners vendors revenue clients dealer transactions entities dealers loansbilities makers popular competitors advertisingries boards businessmen estimates commercial lines advertisements budgets sale properties target buysncies items deliveries metals showing platforms streetps costs competitive policies investor relations debts value cash exportn accounts suppliers claims fund outfits holders share trends listing terminals fields players casinos segments projections spending partnershipstrust commissions store fuels buy facilitiesoa instruments quotes volumes dealings listings increases componentsoutsbacks opportunities marketing producer analysts barrels operators wealth profitmarket offering groups sold chemicals activityeller slides creditors peers concessions giants chains commodity asset utilityg financesil boxes farmers confidence consumptioncies mergers purchase displays"]
    class_three = ["markets chains exporters downgrade strategists brokers aluminum bond firmer mergers forecasting merchandise carriers commodities outlooks downgrades warehouse pork specialty spirits techs trim biotechnology trucking grocery wholesale farms inventory hog apparel modestly expire suppliers pepsi organic pharmacy dated mercantile downward grain brewer lackluster investors greenback industries manipulation controls plummets trades stocks enthusiasm dampened shrimp pet jitters gloomy federated await artificially gains currencies tumble bargains retreat atherogenics producers encouraged chinas poultry dependence volume airplanes sagging trimming peers indexes focusing pontiac seller worrying helsinki biotech caterpillar employment bolstered equities semiconductor beef equity derivatives fuels dips subsidiaries metals gauge tsx mitsui soup gainers retreated trimmed goods bankruptcies stalls cosmetics stoked rand unchanged briefs brightened reassuring refining raw rebates sumitomo automotive pressured mixed cigarette kohl higher alcoa dive weighed plasma spike saks weighs compare pointed insurers dram forex lags tesco aig ratings oecd wine banking oversupply dampen q4 commissions cigarettes steep newspapers dip contractors indicators drinks mirant shop projections 2010 purchases stabilize profitability cereal contributed furniture sector inquiries coke assessment cautious payoffs plunging pullback fare suv bloated slack projects groceries optimism sharply discounts sentiment grocer shrinks globally soda matsushita competitiveness singulus imported sectors beset lender microchips consumers bellwether stuck premium reading institutional mostly asset indicator rural partly turbulence weigh appliances fueled rouse cheer retailing futures pour"]
    class_four = ["computer telescope software computers computing hardware technology system pcer machine it systemsware applications program chip programs server desktop electronic code simulator electronics machines database technologies internet chips infrastructure devices browser device processor tools phone specification application dataulator utility processors laptop architecture spacecraft linux networking platformdget servers superl pcs tool programming microsoft patch engine graphicstor components equipment instrument appicom cpu micro programmer tech drive digitalwall networks servicersorers cards platformsdgets notebook protocol memory semiconductorp satellite keyboard operating 8 packageboard company products languagetation displaysrator lan processingm features rocket robot development security exploit satellitesxjetedot player enterprise game receivers display standard com networkbook probe controller windows instruments installations performance mackis clusters product wireless video multimediacecore robots telescopes sensors project switch source observatory dynamic solution embeddedte phones cyber suite specialized screen solutions mechanisms communication technical popular center unix framework os user vehicle astronomy games telecommunications online client laserier console card printers informationetseron sensor version massivena downloadia implanting info interceptor peripheral players array monitor services pioneer technological vulnerabilityam update designrinoum macintoshnet box proprietaryrs mouse boardsization batteries upgrade booster web developers feature content unitu radioapp media 2 nasatech panel mechanismbot ibm filetime install blade panels analog spaceshipdps core missilemax spaceatics stackamp driveslla wi programmerskit automated pro freeicsac patches component communications portet optical intel flight buffer clusterky researchice"]
    class_four = ["computer software firewall portal specialist antivirus ie capabilities design middleware content appliance application searches keyhole notebooks desktops integration usage mobiles programming feeds hyperion btx device proprietary license labs robot vulnerability tools adapter introducing infringed kit rental sms hacker printer url specifications intelligent custom incorporated toolbar fledged copernic properties unix basic midsize laptops competitor blogging photography intrusion architecture robotic hotmail 11i macintosh feed recorder integrates define location implemented embedded routers functionality upstart grid model listings quietly biz messenger aggressively wordperfect mainframe spoofing plug publishers ventures mouse print 10g vulnerabilities flaws licensed language specifically numerous corporations fee sensor 802 identification revamped surfing consultant interactive multimedia malicious redmond apps promotion manufacturers crm tool solutions suite pcs camcorder prototype chassis reference interoperability ringtone netegrity enterprise trusted recognition operates patches litigation startup mcafee looksmart workstations motherboard built films smartphones storage portfolio alternatives simpler client upgrade norton telescope vendor trend jukebox computing personalization telephony imaging offerings certificate blogger switches treo database integrate collaboration tungsten feature components presentation disposable trusecure install domainkeys ipods sharman bittorrent bug combination mapping innovative patents connectivity distributing cds frequency midrange directories feedster furl microprocessors libraries roaming circulating enhancements acquiring bluetooth environments p2p workplace worms poweredge innovation browsing westbridge downloading hosted networking smb attached dantz handsets rss websphere linksys reliability platform servers virtual user adoption"]
    class_one_encodes = tokenizer(class_one, padding='max_length', truncation=True,)["input_ids"][0]
    class_two_encodes = tokenizer(class_two, padding='max_length', truncation=True,)["input_ids"][0]
    class_three_encodes = tokenizer(class_three, padding='max_length', truncation=True,)["input_ids"][0]
    class_four_encodes = tokenizer(class_four, padding='max_length', truncation=True,)["input_ids"][0]
    
    correct = 0
    total = 0
    for doc, label in tqdm(zip(x, y)):
        one_score = 0
        two_score = 0
        three_score = 0
        four_score = 0
        for word in doc:
            if word == 0:
                continue
            if word in class_one_encodes:
                one_score += 1
            if word in class_two_encodes:
                two_score += 1
            if word in class_three_encodes:
                three_score += 1
            if word in class_four_encodes:
                four_score += 1
            # word = tokenizer.decode([word])
            # if word in class_one:
            #     one_score += 1
            # if word in class_two:
            #     two_score += 1
            # if word in class_three:
            #     three_score += 1
            # if word in class_four:
            #     four_score += 1
            
            scores = [one_score, two_score, three_score, four_score]
        highest_score = np.argmax(scores)
        # print(scores)

        if highest_score == label:
            correct += 1
        total += 1
    print(correct)
    print(total)
    print(correct / total)
    print().shape
    return