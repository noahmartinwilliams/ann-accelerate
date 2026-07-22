module Samps where

import Control.Parallel.Strategies
import Data.Array.Accelerate as A
import qualified Data.ByteString as B
import Data.Function (on)
import Data.List as DL
import Data.List.Split
import GHC.Conc
import Prelude as P
import System.Random
import Types

bs2Answer :: B.ByteString -> [Double]
bs2Answer bs | (B.head bs) P.== 0 =  [0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 1 =  [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 2 =  [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 3 =  [0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 4 =  [0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 5 =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 6 =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 7 =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 8 =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0]
bs2Answer bs | (B.head bs) P.== 9 =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]

combineAnswers :: [Matrix Double] -> Matrix Double
combineAnswers ms = do
    let ls = P.foldr (P.++) [] (P.map (A.toList) ms)
        l = P.length ms
    A.fromList (Z:.10:.l) ls

getSamps :: Int -> Int -> (B.ByteString , B.ByteString) -> [(Matrix Double, Matrix Double)]
getSamps seedNum mbs (imgs, answers) = do
    let splitImgs = bsSplitEvery (28 * 28) imgs
        seed = mkStdGen seedNum
        nums = randoms seed :: [Int]
        imgMats = P.map (bs2Mat mbs) splitImgs
        answerMats = bsSplitEvery 1 answers
        answerMats' = (P.map bs2Answer answerMats)
        shuffled = shuf nums (P.zip imgMats answerMats')
    (Samps.condense mbs shuffled) `using` (parListChunk numCapabilities rdeepseq)

condense :: Int -> [([Double], [Double])] -> [(Matrix Double, Matrix Double)]
condense mbs list = do
    let cs = chunksOf mbs list
    P.map condense' cs where
        condense' :: [([Double], [Double])] -> (Matrix Double, Matrix Double)
        condense' ls = do
            let imgs = P.foldr (P.++) [] (P.map P.fst ls)
                answers = P.foldr (P.++) [] (P.map P.snd ls)
                imgs' = A.fromList (Z:.(28*28):.mbs) imgs
                answers' = A.fromList (Z:.10:.mbs) answers
            (imgs', answers')

shuf :: (P.Ord a) => [a] -> [b] -> [b]
shuf rands els = do
    let zipped = P.zip rands els
        (_, sorted) = P.unzip (sortBy (flip P.compare `on` P.fst) zipped)
    sorted

bsSplitEvery :: Int -> B.ByteString -> [B.ByteString]
bsSplitEvery _ bs | (B.null bs) = []
bsSplitEvery i bs = (B.take i bs ) : (bsSplitEvery i (B.drop i bs))

bs2Mat :: Int -> B.ByteString -> [Double]
bs2Mat mbs bs = do
    let uped = B.unpack bs
        asDoubles = P.map (\x -> P.fromIntegral x :: Double) uped
        scaled = P.map (\x -> (x - 128.0) / 128.0) asDoubles
    scaled

prepareSamp :: Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double)
prepareSamp m = do
    let (l1, l2) = A.unlift m :: (Acc (Matrix Double), Acc (Matrix Double))
    A.lift (A.transpose l1, l2)
