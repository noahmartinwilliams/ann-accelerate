{-# LANGUAGE DeriveGeneric #-}
module Types where

import Control.Monad.Reader
import Control.Monad.State

import Data.Aeson 
import Data.Array.Accelerate as A
import qualified Data.ByteString as B
import GHC.Generics
import ML.ANN.Block
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.IO
import System.Random

data Conf = Conf { inputAF :: String, layers :: String, optimizer :: String, lr :: Double, beta1 :: Double, beta2 :: Double, costF :: String, numEpochs :: Int, miniBatchSize :: Int} deriving(Generic)

data St = St { stBLInfo :: Maybe BLInfo, stTestPhase :: Bool, stStart :: Bool, stTrainFn :: Maybe TrainFn, stTestFn :: Maybe TestFn, stG :: StdGen, stTrainImgs :: [(Matrix Double, Matrix Double)], stTestImgs :: [(Matrix Double, Matrix Double)], stFileToOpen :: String, stOpenFile :: Handle, stFileToWrite :: String, stCloseFile :: Bool} deriving(Generic)
instance ToJSON Conf where
    toEncoding = genericToEncoding defaultOptions

defaultState :: StdGen -> St
defaultState g = St { stBLInfo = Nothing, stTestPhase = False, stStart = True, stTrainFn = Nothing, stTestFn = Nothing, stG = g, stTrainImgs = [], stTestImgs = [], stFileToOpen = "", stOpenFile = stdout, stFileToWrite = "", stCloseFile = False }

instance FromJSON Conf 

getConf :: String -> Maybe Conf
getConf inp = do
    let bs = fromString inp
    Data.Aeson.decode bs :: Maybe Conf

type Mon a = ReaderT Conf (State St) a

data SampFiles = SampFiles { testImgs :: B.ByteString, testAnswers :: B.ByteString, trainImgs :: B.ByteString, trainAnswers :: B.ByteString}

type TrainFn = ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Vector Double, Vector Int, Vector Double))

type TestFn = (Matrix Double -> Matrix Double)
