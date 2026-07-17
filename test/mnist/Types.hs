{-# LANGUAGE DeriveGeneric #-}
module Types where

import Control.Monad.Reader
import Control.Monad.State
import Data.Aeson 
import Data.Array.Accelerate as A
import Data.ByteString
import Data.Map
import GHC.Generics
import ML.ANN.Block
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.IO

data Conf = Conf { inputAF :: String, layers :: String, optimizer :: String, lr :: Double, beta1 :: Double, beta2 :: Double, costF :: String, numEpochs :: Int, miniBatchSize :: Int} deriving(Generic)

instance FromJSON Conf 
instance ToJSON Conf where
    toEncoding = genericToEncoding defaultOptions

data Phase = Start | Train | Test1 | Test2 | Save | Save1 | Done deriving(P.Eq, Show)

data St = St { stPhase :: Phase, stTrainSamps :: [(Matrix Double, Matrix Double)], stTestSamps :: [(Matrix Double, Matrix Double)], stBLInfo :: BLInfo, stBlock :: (Vector Int, Vector Double), stFilesToWrite :: Map String ByteString, stFilesToOpen :: [String], stFiles :: Map String Handle, stFilesToClose :: [String] }

type Mon = ReaderT Conf (State St) 

type TrainFn = ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Vector Double, (Vector Int, Vector Double)))

type TestFn = (Matrix Double -> Matrix Double)

type SampSource = (ByteString, ByteString, ByteString, ByteString)
