require(tidyverse)
require(arrow)
require(skimr)
require(gganimate)
require(units)

base_dir <- "~/Dropbox/NFL/Big Data Bowl/"

#### somehow combine tracking files ####

tracking_files <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "weeks",full.names = T)) %>% 
  map(~list.files(.x,full.names = T))

schema1 <- schema(time = timestamp(unit = "ns",timezone = "GMT"),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),dir = float64(),
                  event = string(),nflId = int64(),displayName = string(),jerseyNumber = int16(),position = string(),
                  frameId = int16(),team = string(),gameId = int32(),playId = int32(),
                  playDirection = string(),route = string())

schema2 <- schema(gameId = int32(),playId = int32(), nflId = int64(), frameId = int16(),
                  time = timestamp(unit = "ns"),jerseyNumber = int16(),team = string(),playDirection = string(),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
                  dir = float64(),event = string())

schema3 <- schema(gameId = int32(),playId = int32(), nflId = int64(), displayName = string(), frameId = int16(),
                  frametype = string(),time = timestamp(unit = "ns"),jerseyNumber = int16(),club = string(),
                  playDirection = string(),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
                  dir = float64(),event = string())

ds1 <- open_dataset(tracking_files[[1]],format = "csv",schema = schema1,skip = 1)
ds2 <- open_dataset(tracking_files[[2]],format = "csv",schema = schema2,skip = 1)
ds3 <- open_dataset(tracking_files[[3]],format = "csv",schema = schema3,skip = 1)

ds <- ds1 %>% 
  select(all_of(names(ds2))) %>% 
  select(-time) %>% 
  union_all(
    ds2 %>% 
      select(-time)
  ) %>% 
  union_all(
    ds3 %>% 
      rename(team = club) %>% 
      select(all_of(names(ds2))) %>% 
      select(-time)
  )

ds %>% 
  filter(event != "NA") %>% 
  count(event) %>% 
  collect

events_to_look_at <- c("ball_snap","pass_forward","qb_sack","qb_strip_sack",
                       "autoevent_passforward","autoevent_ballsnap",
                       "autoevent_passinterrupted","pass_shovel","fumble")

sig_events <- ds %>% 
  filter(event %in% events_to_look_at) %>% 
  distinct(gameId,playId,frameId,event) %>% 
  collect

sig_events %>% 
  list(
    filter(.,event %in% c("ball_snap","autoevent_ballsnap")) %>% 
      group_by(gameId,playId) %>% 
      summarise(snap = min(frameId),
                .groups = "drop")
  ) %>% 
  reduce(left_join,by = c("gameId","playId")) %>% 
  mutate(frame_diff = frameId - snap) %>% 
  filter(!(event %in% c("ball_snap","autoevent_ballsnap"))) %>% 
  filter(frame_diff < 100) %>% 
  ggplot(mapping = aes(x = frame_diff,fill = event))+
  geom_density(alpha = .7)+
  theme_gray()
