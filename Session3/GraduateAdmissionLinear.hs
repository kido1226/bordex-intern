module GraduateAdmissionLinear (runGraduateAdmissionLinear) where

{-# LANGUAGE DeriveGeneric #-}
import Control.Monad (zipWithM_)
import Torch.Functional as F
import Torch.Tensor (Tensor, asTensor, asValue)

import Data.Text (Text)
import GHC.Generics (Generic)

import qualified Data.ByteString.Lazy as BL
import Data.Char (ord)
import Data.Csv
import qualified Data.Vector as V

data Train = Train
  { x :: !Float
    y :: !Float
  }
  deriving (Generic, Show)

instance FromNamedRecord Train

mkMsg :: Train -> [Tensor, Tensor]
mkMsg tr =
    name p
    <> " earns "
    <> show (salary p)
    <> " dollars"

runGraduateAdmissionLinear :: IO ()
runGraduateAdmissionLinear = do
    csvData <- BL.readFile "data/train.csv"
    print csvData