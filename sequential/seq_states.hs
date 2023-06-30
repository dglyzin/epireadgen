import Control.Monad.State
import Control.Monad.Writer
import System.Random
import Data.Ratio
import Data.List
import Data.Foldable
import System.IO


randFloat::(RandomGen g)=>State g Float 
randFloat = state (randomR (0.0, 1.0))

randInt::(RandomGen g)=>State g Int 
randInt = state (randomR (0, 1))

type RandSeq gen a = StateT [a] (State gen) a

eff_st:: (RandomGen g, Random a)=>State g a->RandSeq g a
--eff_st:: (RandomGen g, Random a)=>State g a->StateT [a] (State g) a
eff_st rand = do
  r <- lift $ rand
  s <- get
  put (s++[r])
  return r


-- The sum of n independent random variables Xi
sumXi::(RandomGen g, Num a)=>Int->RandSeq g a->RandSeq g a
-- sumXi::(RandomGen g, Num a)=>Int->StateT [a] (State g) a->StateT [a] (State g) a
-- sumXi n s0= foldr (\acc x->(+)<$>acc<*>x) s0 (replicate n s0)
sumXi n s0= foldr wrapper s0 (replicate n s0)
 where
   wrapper::(RandomGen g, Num a)=>RandSeq g a->RandSeq g a-> RandSeq g a
   -- wrapper::(RandomGen g, Num a)=>StateT [a] (State g) a->StateT [a] (State g) a-> StateT [a] (State g) a
   wrapper acc x = do

     -- init values:
     let tmp_state = []
     gen<-(lift $ get)

     -- main:
     let (r, gen1) = (runState (evalStateT ((+)<$>acc<*>x) tmp_state) gen)

     -- adding result to the history:
     s<-get
     put (s++[r])
     
     -- updating random generator:
     lift $ put gen1
     return r


type WRandSeq gen a = WriterT [String] (StateT [a] (State gen)) a
eff_w::(RandomGen g, Random a)=>State g a->WRandSeq g a
eff_w rand = do
  r <- lift $ lift $ rand
  s <- lift $ get
  lift $ put (s++[r])
  tell ["ho"]
  return r


wSumXi::(RandomGen g, Num a, Show a)=>Int->WRandSeq g a-> WRandSeq g a
wSumXi n ws0 = foldr wrapper ws0 (replicate n ws0)
  where
    wrapper::(RandomGen g, Num a, Show a)=>WRandSeq g a-> WRandSeq g a-> WRandSeq g a
    wrapper acc x = do
     s<-(lift $ get)
     -- init values:
     --let tmp_state = []
     -- gen<-(lift $ lift $ get)
     acc_val <- acc
     
     x_val <- x
  
     -- let (r, gen1) = (runState (evalStateT ((\ x y -> ((+)<$>x<*>y))<$>sT_acc<*>sT_x) tmp_state) gen)

     -- adding result to the history:

     lift $ put (s++[acc_val+x_val])

     -- updating random generator:
     -- lift $ lift $ put gen1

     tell [(show s)]
     return (acc_val+x_val)

{-
wSumXi'::(RandomGen g)=>Int-> WriterT [String] (StateT [Int] (State g)) ()
wSumXi' n = traverse_ (\i-> (wrapper randInt)) [1..n]
  where
    wrapper::(RandomGen g)=>State g Int->WriterT [String] (StateT [Int] (State g)) ()
    wrapper rand = do
      
      let 
      --xi <- lift $ lift $ rand
      --g0 <- lift $ lift $ get
      -- fxi <- lift $ lift $ return
      let (xi, gen)= 
      --lift $ lift $ put g0
      -- xi <- runWriterT
      --s <- lift $ get
      let s' = (case s of
                  [] -> [xi]
                  s11@(s1:rest) -> [xi+s1]++s11)
      lift $ put s'
      tell [(show xi)]
      return ()
-}

main = do
  
  let gen = (mkStdGen 2023)
  (putStr "test5: wSumXi1:\n")
  --(putStr $ (show (runState (runStateT (runWriterT (wSumXi' 3)) []) gen)))
  (putStr "\n----------------\n")
  
  (putStr "test5: wSumXi:\n")
  (putStr $ (show (runState (runStateT (runWriterT (wSumXi 3 (eff_w randFloat))) []) gen)))
  (putStr "\n----------------\n")

  (putStr "test4: writer:\n")
  (putStr $ (show (runState (runStateT (runWriterT (eff_w randFloat)) []) gen)))
  (putStr "\n----------------\n")
  
  (putStr "test3: the sum S of independ. rand. vars Xi:\n")
  let (s, gen1) = (runState (execStateT (traverse (\n->(sumXi 3 (eff_st randInt))) [1..3]) []) gen)
  (putStr $ (show s))
  -- (putStr $ (show (runState (runStateT (sumXi 3 eff_st) []) gen)))
  -- (putStr $ (show (runState (runStateT (sumXi 3 eff_st) []) gen)))
  -- (putStr $ (show (runState (runStateT ((+)<$>eff_st<*>eff_st) []) gen)))
  (putStr "\n----------------\n")
  
  (putStr "test2: traversing of the rand state transformer:\n")
  (putStr "test2: n times sum or tree vars:\n") 
  let (s, gen1) = (runState (execStateT (traverse (\n->(eff_st randFloat)) [1..3]) []) gen)
  (putStr $ (show s))
  --(putStr $ (show (runState (runStateT (traverse (\n->eff_st) [1,2,3]) []) gen)))
  (putStr "\n----------------\n")

  (putStr "test1: the rand state transformer:\n") 
  (putStr $ (show (runState (runStateT (eff_st randFloat) []) gen)))
  (putStr "\n----------------\n")

  (putStr "test0: the rand state:\n") 
  let (v, g) = (runState randFloat (mkStdGen 2023))
  -- (putStr $ (show v))
  let (v1, g1) = (runState randFloat g)
  -- (putStr $ (show v1))
  (putStr $ unlines . map show $ [v, v1])
