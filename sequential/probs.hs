import Data.Ratio
import System.Random
import Data.List
import Control.Monad.State
import System.IO
import Data.Time.Clock.System
import Data.Foldable

--extending type [(a, Rational)]:
-- FOR this first example of using prob is taking from forGreatGood
-- here just for understanding the concept and also make refs to the good book:
newtype Prob a = Prob {getProb::[(a, Rational)]} deriving Show
{-
*Main> Prob [("x", 3%4)]
Prob {getProb = [("x",3 % 4)]}
*Main> Prob {getProb=[(1, 1%3)]}
Prob {getProb = [(1,1 % 3)]}
-}
instance Functor Prob where
  fmap f (Prob xs) = Prob $ map (\(x, p)->(f x, p)) xs

-- ghci> fmap negate (Prob [(3,1%2),(5,1%4),(9,1%4)])

flatten:: Prob (Prob a) -> Prob a
flatten (Prob xs) =
  Prob $ join $ map (\(vs, p)-> map (\(v, p1)-> (v, p1*p)) (getProb vs)) xs
  where join::[[a]]->[a]
        join (x:xs) = x++join xs
        join [] = []

sim_box::Prob (Prob Char)
sim_box = Prob [(Prob [('a', 1%2), ('b', 1%2)], 1%4),
                (Prob [('c', 2%3), ('d', 1%3)], 3%4)]
-- END FOR

-------------------------------------
-- this is my actual implementation of aima weighted update
type Var = String

type Event = (Var, [Var])
-- value with parent
type PVal a = (a, [a])
type CondProb a = Prob (a, [a])
data Event1 a = Event1 {eget_var::Var, eget_evidence::[Var], eget_support::[a], eget_prob::CondProb a} deriving Show
{-TESTS:
        *Main> let e = Event1 {eget_var="x", eget_evidence=["y","z"], eget_support=[0,1,2], eget_prob=(Prob [((0,[0,0]),1%3), ((0, [0,0]),2%3), ((1,[0,0]),1%3), ((1, [0,0]),2%3)])}
        Main> eget_var e
        "x"
        *Main> eget_evidence e
        ["y","z"]
        *Main> eget_prob e
        Prob {getProb = [((0,[0,0]),1 % 3),((0,[0,0]),2 % 3)]}
        *Main> getProb $ eget_prob e
        [((0,[0,0]),1 % 3),((0,[0,0]),2 % 3)]

        *Main System.Random Data.Ratio> let e = Event1 {eget_var="x", eget_evidence=["y","z"], eget_support=[0,1], eget_prob=(Prob [((0,[0,0]),1%3), ((1, [0,0]),2%3), ((0,[0,1]),1%2), ((1, [0,1]),1%2), ((0, [1, 0]), 1%4), ((1, [1,0]), 3%4), ((0, [1,1]), 1%6), ((1, [1,1]), 5%6)])}
-}

ez::(Num a)=>Event1 a
ez = Event1 {eget_var="z", eget_evidence=[], eget_support=[0, 1], eget_prob=(Prob [((0,[]),1%3), ((1, []),2%3)])}

ey::(Num a)=>Event1 a
ey = Event1 {eget_var="y", eget_evidence=["z"], eget_support=[0, 1], eget_prob=(Prob [((0,[0]),1%2), ((1, [0]),1%2), ((0,[1]),1%3), ((1, [1]),2%3)])}

ex::(Num a)=>Event1 a
ex = Event1 {eget_var="x", eget_evidence=["y","z"], eget_support=[0,1], eget_prob=(Prob [((0,[0,0]),1%3), ((1, [0,0]),2%3), ((0,[0,1]),1%2), ((1, [0,1]),1%2), ((0, [1, 0]), 1%4), ((1, [1,0]), 3%4), ((0, [1,1]), 1%6), ((1, [1,1]), 5%6)])}


data BayesNet a = BayesNet {getBNet::[Event1 a]} deriving Show
{-TESTS:
        *Main> BayesNet [e]
        BayesNet {getBNet = [Event1 {var = "x", evidence = ["y","z"], support=[0], prob = Prob {getProb = [((0,[0,0]),1 % 3),((0,[0,0]),2 % 3)]}}]}

        *Main> let ey = Event1 {eget_var="y", eget_evidence=["z"], eget_support=[0, 1], eget_prob=(Prob [((0,[0]),1%2), ((1, [0]),1%2)])}
        *Main> let ez = Event1 {eget_var="z", eget_evidence=[], eget_support=[0, 1], eget_prob=(Prob [((0,[]),1%3), ((1, []),2%3)])}
        *Main> BayesNet [e,ey,ez]
-}

-----------------------
to_event::Event1 a -> Event
to_event e = (eget_var e, eget_evidence e)

find_event:: [Event1 a]->Event -> Maybe (Event1 a)
find_event (e:es) qe@(name, evidence)
  | eget_var e == name = Just e
  | otherwise = find_event es qe

find_event [] (name, _) = Nothing
{-TESTS:
        *Main> find_event (getBNet $ BayesNet [e,e1,ez]) ("x",[])
-}
  
-----------------------
-- for zipper for storing and preserve events
type BZipper = (Event, [Var])
get_to_top::BZipper -> Event
get_to_top ((x, es), b:bs) = get_to_top ((x, [b]++es), bs)
get_to_top ((x, es), []) = (x, es)

-- separate events at (that contained in parents of e or precede to that which contained, that not)
epush::Event1 a->[Event1 a]->[Event1 a]
epush e es = let (les, res) = espan [] (eget_evidence e) es in
               les++[e]++res
{-Tests:
*Main> map eget_var $ epush ex [ey]
["y","x"]
*Main> (eget_evidence ex)
["y","z"]
*Main> map eget_var $ epush ex [ez, ey]
["z","y","x"]
*Main> map eget_var $ epush ex [ey, ez]
["y","z","x"]

--when there is no parents to satisfy, it put to the last
*Main> eget_evidence ey
["z"]

*Main> map eget_var $ epush ey [ex, ex, ex]
["x","x","x","y"]

--when there  is no parents at all:
*Main> eget_evidence ez
[]

*Main> map eget_var $ epush ez [ex, ex, ex]
["z","x","x","x"]

-- until all parents present:
*Main> eget_evidence ey
["z"]

*Main> map eget_var $ epush ey [ex, ex, ex, ez, ex]
["x","x","x","z","y","x"]

*Main> eget_evidence ex
["y","z"]

*Main> map eget_var $ epush ex [ex,  ez, ex, ey, ex, ey]
["x","z","x","y","X","x","y"]

-}

espan:: [Event1 a]->[Var]->[Event1 a]->([Event1 a], [Event1 a])
espan les [] (e:res) = (les, ([e]++res))
espan les ps [] = (les, [])
espan les ps (e:res) = 
      let (ls, rs) = pspan [] (eget_var e) ps in
          espan (les++[e]) (ls++rs) res

-- separate parents
pspan::[Var]->Var->[Var]->([Var], [Var])
pspan ls elm (r:rs) 
     | elm == r = (ls, rs)
     | otherwise = pspan (ls ++ [r]) elm rs
pspan ls elm [] = (ls, [])

{-- for pushing to the ordered BayesNet
-- tests:
*Main> let ps = [("z1", ["z0"]), ("y",["z1"])]
*Main> push (("x", ["y", "z1"]), []) ps
[("z1",["z0"]),("y",["z1"]),("x",["y","z1"])]
-}
push::BZipper->[Event]->[Event]
push ((x, e:es), bs) ((x', es'):vs)
     | e == x' = [(x', es')] ++ push ((x, es), [e]++bs) vs
     | otherwise = [(x', es')] ++ push ((x, [e]++es), bs) vs
push ((x, []), bs) rs
     = rs ++ [get_to_top ((x, []), bs)]
push zp [] = [get_to_top zp]


-------------
type Sample a = (Var, a) -- ("x", 0) or ("x", (0, [1, 0])) if there is parents

type Obs a = [Sample a]
--oplus::Obs a->Obs a -> Obs a
get_obs_val:: Var->Obs a-> Maybe a
get_obs_val x ((o, val):obs)
  | x == o = Just val
  | otherwise = get_obs_val x obs
get_obs_val x [] = Nothing


type Weight a b = (Sample a, b)
type WZipper a b = ([Weight a b], [Weight a b])
wget_to_top::WZipper a b->[Weight a b]
wget_to_top (ws, b:bs) = wget_to_top ([b]++ws, bs)
wget_to_top (ws, []) = ws

-- TODO: a->State ws w 
-- (Sample a, w)-> modify (\ws->update ws (Sample a, w))
-- update an weight of the (var,val) in weights list and return it:
wupdate::(Eq a, Num b)=>[Weight a b]->Sample a->b->[Weight a b]
wupdate ws s w = wupdate1 (ws, []) s w

wupdate1::(Eq a, Num b)=> WZipper a b->Sample a->b->[Weight a b]
wupdate1 ((((var', val'), w'):ws), bs) (var, val) w
  | var'==var && val==val' = wget_to_top ([((var', val'), w'*w)]++ws, bs)
  | otherwise = wupdate1 (ws, [wnode]++bs) (var, val) w
  where
    wnode = ((var', val'), w')
wupdate1 ([], bs) s w = wget_to_top ([], bs)
{-
Tests:
 *Main> let weights = [(("Color", "red"), 1), (("Color", "blue"), 0), (("Box", "0"), 3)]

 *Main> wupdate weights ("Color", "blue") 2
    [(("Color","red"),1),(("Color","blue"),2),(("Box","0"),3)]

 *Main> wupdate weights ("Color", "red") 2
    [(("Color","red"),3),(("Color","blue"),0),(("Box","0"),3)]
-}

-- init weights:
init_weights::(Eq a, Num b)=>[Event1 a] -> [Weight (a, [a]) b]
-- event a-> ((var_name, var_val), 1)
init_weights es = 
       es >>= (\e-> map (\val->((eget_var e, fst val), 1))
                        (getProb $ eget_prob e))
{-Tests:
init_weights [ez, ey, ex]
[(("z",(0,[])),0),(("z",(1,[])),0),(("y",(0,[0])),0),(("y",(1,[0])),0),(("y",(0,[1])),0),(("y",(1,[1])),0),(("x",(0,[0,0])),0),(("x",(1,[0,0])),0),(("x",(0,[0,1])),0),(("x",(1,[0,1])),0),(("x",(0,[1,0])),0),(("x",(1,[1,0])),0),(("x",(0,[1,1])),0),(("x",(1,[1,1])),0)]
-}


-- TODO: to StateT Weights (StateT Obs (State g)) a
weighted_sample::(Eq a, RandomGen g)=>[Event1 a]->Obs (a, [a])->[Weight (a, [a]) Rational]->g->((Obs (a, [a]), [Weight (a, [a]) Rational]), g)
-- look at the event then find a probability of what are you looking at then add it to the weigths
-- BUG: updating weights happend in all case rather then only for the evidence vars!
weighted_sample [] obs ws gen = ((obs, ws), gen)
weighted_sample (e:es) obs ws gen = 
           let ( (obs', gen'), sample) = observe (obs, gen) e in
               let probability = (get_prob (eget_prob e) (snd sample)) in
                   weighted_sample es obs' (wupdate ws sample probability) gen'
{-
Tests:

*Main> init_weights [ez]
[(("z",(0,[])),1),(("z",(1,[])),1)]

let ((obs, ws), gen) = weighted_sample [ez] [] (init_weights [ez]) (mkStdGen 2023)
*Main Control.Applicative> obs
[("z",(1,[]))]
*Main Control.Applicative> ws
[(("z",(0,[])),1 % 1),(("z",(1,[])),2 % 3)]


let ((obs, ws), gen) = weighted_sample [ez, ey] [] (init_weights [ez, ey]) (mkStdGen 2023)
*Main Control.Applicative> obs
[("z",(1,[])),("y",(1,[1]))]
*Main Control.Applicative> ws
[(("z",(0,[])),1 % 1),(("z",(1,[])),2 % 3),(("y",(0,[0])),1 % 1),(("y",(1,[0])),1 % 1),(("y",(0,[1]vv)),1 % 1),(("y",(1,[1])),2 % 3)]

let ((obs, ws), gen) = weighted_sample [ez, ey, ex] [] (init_weights [ez, ey, ex]) (mkStdGen 2023)
*Main Control.Applicative> obs
[("z",(1,[])),("y",(1,[1])),("x",(1,[1,1]))]
 ws
[(("z",(0,[])),1 % 1),(("z",(1,[])),2 % 3),(("y",(0,[0])),1 % 1),(("y",(1,[0])),1 % 1),(("y",(0,[1])),1 % 1),(("y",(1,[1])),2 % 3),(("x",(0,[0,0])),1 % 1),(("x",(1,[0,0])),1 % 1),(("x",(0,[0,1])),1 % 1),(("x",(1,[0,1])),1 % 1),(("x",(0,[1,0])),1 % 1),(("x",(1,[1,0])),1 % 1),(("x",(0,[1,1])),1 % 1),(("x",(1,[1,1])),5 % 6)]

# or

> let ((obs, ws), gen) = weighted_sample [ez] [] (init_weights [ez, ey, ex]) (mkStdGen 2023)
> let ((obs', ws'), gen') = weighted_sample [ez, ey] obs ws gen
> obs'
[("z",(1,[])),("y",(1,[1]))]
-- note that the "y" weight increased in ws' according to obs'
*Main Control.Applicative> ws'
[(("z",(0,[])),1 % 1),(("z",(1,[])),4 % 9),(("y",(0,[0])),1 % 1),(("y",(1,[0])),1 % 1),(("y",(0,[1])),1 % 1),(("y",(1,[1])),2 % 3),(("x",(0,[0,0])),1 % 1),(("x",(1,[0,0])),1 % 1),(("x",(0,[0,1])),1 % 1),(("x",(1,[0,1])),1 % 1),(("x",(0,[1,0])),1 % 1),(("x",(1,[1,0])),1 % 1),(("x",(0,[1,1])),1 % 1),(("x",(1,[1,1])),1 % 1)]

> let ((obs'', ws''), gen'') = weighted_sample [ez, ey, ex] obs' ws' gen'
*Main Control.Applicative> obs''
[("z",(1,[])),("y",(1,[1])),("x",(1,[1,1]))]

-- note that the "x" weight increased in ws'' according to obs''
*Main Control.Applicative> ws''
[(("z",(0,[])),1 % 1),(("z",(1,[])),8 % 27),(("y",(0,[0])),1 % 1),(("y",(1,[0])),1 % 1),(("y",(0,[1])),1 % 1),(("y",(1,[1])),4 % 9),(("x",(0,[0,0])),1 % 1),(("x",(1,[0,0])),1 % 1),(("x",(0,[0,1])),1 % 1),(("x",(1,[0,1])),1 % 1),(("x",(0,[1,0])),1 % 1),(("x",(1,[1,0])),1 % 1),(("x",(0,[1,1])),1 % 1),(("x",(1,[1,1])),5 % 6)]

-- it can be used recursevly with a new obs [] but the old weights ws'':
> let ((obs''', ws'''), gen''') = weighted_sample [ez, ey, ex] [] ws'' gen''
*Main Control.Applicative> obs'''
[("z",(1,[])),("y",(1,[1])),("x",(1,[1,1]))]
*Main Control.Applicative> ws'''
[(("z",(0,[])),1 % 1),(("z",(1,[])),16 % 81),(("y",(0,[0])),1 % 1),(("y",(1,[0])),1 % 1),(("y",(0,[1])),1 % 1),(("y",(1,[1])),8 % 27),(("x",(0,[0,0])),1 % 1),(("x",(1,[0,0])),1 % 1),(("x",(0,[0,1])),1 % 1),(("x",(1,[0,1])),1 % 1),(("x",(0,[1,0])),1 % 1),(("x",(1,[1,0])),1 % 1),(("x",(0,[1,1])),1 % 1),(("x",(1,[1,1])),25 % 36)]
-}

-- TODO(stateful): (e, obs)->(val, (e, obs))
-- switch from (Obs, e) -> (Obs, sample) 
-- to  ((as_sample e)->sample)<$>(Obs . Sample)
--    StateT [Obs.Sample] (State g) a

-- `Sample a` - The form of the `a` here is (val, parents_vals) like (0, [0, 0]) 
-- so `Sample a` means (var, (val, parents_vals)) like ("x", (0, [0, 0]))
type SuccObs g a = StateT (Obs (a, [a])) (State g)
observe'::(RandomGen g, Eq a)=>Event1 a -> SuccObs g a (Sample (a, [a]))
observe' e = do
  obs <- get
  case get_obs_val (eget_var e) obs of
    Just val -> return (eget_var e, val)
    Nothing -> do
      let parents_vals = get_parents (eget_evidence e) obs
      sample<-lift $ (to_sample' e parents_vals)
      put (obs++[sample])
      return sample

observe::(RandomGen g, Eq a)=> (Obs (a, [a]), g)->Event1 a->((Obs (a, [a]), g), Sample (a, [a]))
observe (obs, gen) e = 
      case get_obs_val (eget_var e) obs of 
           Just val -> ((obs, gen), (eget_var e, val)) 
           Nothing -> let parents_vals = get_parents (eget_evidence e) obs in
                         let (sample, gen') = (to_sample e parents_vals gen) in
                            ((obs++[sample], gen'), sample)
{-
Tests:
*Main> let ((obs, gen), sample) = observe ([], (mkStdGen 2021)) ez
*Main> let ((obs', gen'), sample') = observe (obs, gen) ey
*Main> let ((obs'', gen''), sample'') = observe (obs', gen') ex
*Main> obs''
[("z",(1,[])),("y",(1,[1])),("x",(1,[1,1]))]

# old:
*Main> let (sample, obs, gen) = observe [] ez (mkStdGen 2021)
*Main> let (sample', obs', gen') = observe obs ey gen
*Main> let (sample'', obs'', gen'') = observe obs' ex gen'
*Main> obs''
[("z",(1,[])),("y",(1,[1])),("x",(1,[1,1]))]
-}
      
get_parents:: [Var]->Obs (a, [a])->[a]
get_parents ps obs = get_parents1 ps obs []

get_parents1:: [Var]-> Obs (a, [a])->[Sample (a, [a])]->[a]
get_parents1 ps [] bs = []
get_parents1 [] obs bs = []
get_parents1 ps@(pvar:ps_rest) (o@(var,(val, _)):obs_rest) bs
     -- if found it go to search next parent in bs++obs_rest:
     | pvar == var = [val]++(get_parents1 ps_rest (bs++obs_rest) [])

     -- else remember o and go check next obs
     | otherwise = get_parents1 ps obs_rest (bs++[o])
{-
Tests:
*Main> get_parents ["y", "z"] [("k", (0,[0,3])), ("y", (1, [1])), ("z", (2, []))]
[1,2]

Main> get_parents ["y", "z"] [("k", (0,[0,3])), ("z", (1, [1])), ("y", (2, []))]
[2,1]

*Main> get_parents ["z", "y"] [("k", (0,[0,3])), ("z", (1, [1])), ("y", (2, []))]
[1,2]

*Main> get_parents ["z", "y"] [("k", (0,[0,3])), ("z", (1, [1]))]
[1]
*Main> get_parents ["z", "y"] [("k", (0,[0,3]))]
[]
-}

   
-- to sample from event given parents ps and gen
to_sample'::(RandomGen g, Eq a)=>Event1 a->[a]->State g (Sample (a, [a]))
to_sample' event ps = do
  random_number <- rand 
  let f = (\acc (var, p)-> if (sum $ map (fromRational . snd) acc) + (fromRational (p))<=(random_number) then acc++[(var, p)] else acc)
  -- eget_prob will return Prob [((0, [0,0]), 1%3), ((1, [0, 0]), 2%3)]
  -- head and tail separation for working with case when random_number< min(probs)
  let (first_entry:probs_table_tail) = (get_probs (eget_prob event) ps)
  let sampled_val = fst $ last $ foldl f [first_entry] probs_table_tail 
  return (eget_var event, sampled_val)
         
to_sample::(RandomGen g, Eq a)=>Event1 a->[a]->g->(Sample (a, [a]), g)
to_sample event ps gen = 
          let (random_number, gen') = (rand gen) in 
              let f = (\acc (var, p)-> if (sum $ map (fromRational . snd) acc) + (fromRational (p))<=(random_number) then acc++[(var, p)] else acc) in
                 -- eget_prob will return Prob [((0, [0,0]), 1%3), ((1, [0, 0]), 2%3)]
                 -- head and tail separation for working with case when random_number< min(probs)
                 let (first_entry:probs_table_tail) = (get_probs (eget_prob event) ps) in
                     let sampled_val = fst $ last $ foldl f [first_entry] probs_table_tail in 
                         ((eget_var event, sampled_val), gen')
          where 
              rand::(RandomGen g)=>g->(Float, g) 
              rand gen = randomR (0.0, 1.0) gen

{-
tests:
  Main System.Random Data.Ratio Data.List> let e1 = Event1 {eget_var="x", eget_evidence=["y"], eget_support=[0, 1, 2, 3], eget_prob=(Prob [((0,[0]),1%4), ((1, [0]),1%4), ((2,[0]),1%4), ((3, [0]),1%4), ((0, [1]), 1%8), ((1, [1]), 1%8), ((2, [1]), 5%8), ((3, [1]), 1%8)])}

  *Main System.Random Data.Ratio> fst $ to_sample e1 [1] (mkStdGen 131) 
  ("x",(2,[1]))

  *Main System.Random Data.Ratio Data.List> map (\((v, ps),p)->(v, p)) $ get_probs (eget_prob e1) [1]
  [(2,5 % 8),(0,1 % 8),(1,1 % 8),(3,1 % 8)]

  *Main System.Random Data.Ratio Data.List> map (fst . snd) $ take 10 $  unfoldr (\(gen) -> Just (to_sample e1 [1] gen)) (mkStdGen 2021)
  [2,2,2,0,2,2,2,2,2,2]
-}

rand::(RandomGen g)=>State g Float
rand = state (randomR (0.0, 1.0))
{-
Tests:
*Main System.Random> (runState $ rand) (mkStdGen 2023)
(0.19624233,114580137 1655838864)

-}

-- to get the `probs` table from the given probabilities `probs` satisfying the parents values `ps`
get_probs::(Eq a)=>CondProb a->[a]->[((a, [a]), Rational)]
get_probs probs ps = sortBy (\(x1, p1) (x2, p2)-> compare p2 p1) $ filter (\((val, pvals), p)->(pvals==ps)) (getProb probs) 
{-
tests:
   *Main System.Random Data.Ratio Data.List> get_probs (eget_prob e1) [0] 
   [((0,[0]),1 % 4),((1,[0]),1 % 4),((2,[0]),1 % 4),((3,[0]),1 % 4)]

   *Main System.Random Data.Ratio Data.List> get_probs (eget_prob e1) [1] 
   [((2,[1]),5 % 8), ((0,[1]),1 % 8),((1,[1]),1 % 8),((3,[1]),1 % 8)]

-}
get_prob::(Eq a)=>Prob a->a->Rational
get_prob probs s = _get_prob (getProb probs) s
    where
       _get_prob::(Eq a)=>[(a, Rational)]->a->Rational
       _get_prob ((a, p):rest) s 
              | a == s = p
              | otherwise = _get_prob rest s
       --if nothing defined should be an error:
       _get_prob [] s = 0 

-- TODO:
--likelihood-weighting:: Var->Obs a->BayesNet a->Int->a
--likelihood-weighting x ((o, val):obs) e:es n

-- TODO: use `rec_do` from `state.hs` to implement  rand, to_sample,  observe, weighted-sample, wupdate

succ_seq::(RandomGen g, Eq a)=>SuccObs g a (Sample (a, [a])) ->StateT Int (SuccObs g a) ()
succ_seq succ = do
  a<-lift $ succ
  return ()

-- and, finally, the sotch. process, generated by StateT transformer:
type SeqState' g a = StateT Int (SuccObs g a)
--succ_seq = state (\((((a, est), sumst), jst), gen)->run_once succ_ehandler est sumst jst gen)

succ_seq1::(RandomGen g, Eq a)=> [Event1 a] -> SeqState' g a ()
succ_seq1 = traverse_ (\e-> succ_seq (observe' e))

main = do
  time <- getSystemTime
  let gen = (mkStdGen (read (show (systemNanoseconds time))))

  (putStr "test3: succ_seq1' [ez, ey, ex]::\n")
  (putStr $ (show (runState (runStateT (runStateT (succ_seq1 [ez, ey, ex]) 0) []) gen)))
  (putStr "\n----------------\n")

  (putStr "test2: observe' ex::\n")
  (putStr $ (show (runState (runStateT (observe' ez) []) gen)))
  (putStr "\n----------------\n")

  (putStr "test1: to_sample' ex [ey, ez]::\n")
  (putStr $ (show (runState (to_sample' ex [0, 1]) gen)))
  (putStr "\n----------------\n")

