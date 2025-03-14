coverages_to_model = c(
  "Cover-0",
  "Cover-1",
  "Cover-2",
  "Cover-3",
  "Quarters",
  "Cover-6"
)

snap_events <- c("ball_snap","autoevent_ballsnap","snap_direct")

events_to_clip <- c(
  'pass_forward','autoevent_passforward','qb_sack','run','pass_outcome_incomplete','pass_arrived',
  'autoevent_passinterrupted','qb_strip_sack','pass_tipped','fumble','pass_outcome_caught','handoff',
  'pass_shovel','qb_spike','pass_outcome_interception', 'lateral', 'tackle',
  'fumble_defense_recovered', 'fumble_offense_recovered', 'qb_slide',
  'pass_outcome_touchdown', 'dropped_pass', 'safety'
)

play_crop <- ds %>% 
  distinct(gameId,playId,frameId,event) %>% 
  group_by(gameId,playId) %>% 
  mutate(
    snap_event = event %in% snap_events,
    clip_event = event %in% events_to_clip
  ) %>% 
  list(
    filter(.,snap_event == T),
    filter(.,clip_event == T)
  ) %>% 
  .[-1] %>% 
  map(function(x) {
    
    group_by(x,gameId,playId) %>% 
      summarise(across(frameId,min),.groups = "drop") %>% 
      collect
    
  }) %>% 
  list(c("snap","clip")) %>% 
  pmap(function(x,y) mutate(x,type = y)) %>% 
  bind_rows %>% 
  pivot_wider(id_cols = c(gameId,playId),names_from = type,values_from = frameId)

spots <- ds %>% 
  filter(team == "football") %>% 
  mutate(across(c(playId,gameId,frameId),as.integer)) %>% 
  inner_join(
    play_crop %>% 
      rename(frameId = snap) %>% 
      mutate(across(frameId,as.integer)),
    by = c("gameId","playId","frameId")
  ) %>% 
  select(gameId,playId,spot_x = x,spot_y = y) %>% 
  collect

mean_positions <- ds %>% 
  left_join(
    play_crop,
    by = c("gameId","playId")
  ) %>% 
  filter(frameId >= snap & (frameId <= clip | is.na(clip))) %>% 
  list(
    plays_clean %>% 
      mutate(across(c(gameId,playId),as.integer)) %>% 
      select(gameId,playId,defensiveTeam,yardsToGo)
  ) %>% 
  reduce(inner_join,by = c("gameId","playId")) %>% 
  filter(team == defensiveTeam) %>% 
  left_join(
    spots,
    by = c("gameId","playId")
  ) %>% 
  mutate(
    flip_direction = ifelse(playDirection == "right","counterclockwise","clockwise"),
    newx = ifelse(flip_direction == "counterclockwise", y, -y + 53.33),
    newy = ifelse(flip_direction == "counterclockwise", -x, x),
    newo = ifelse(flip_direction == "counterclockwise",-(o +90), -(o - 90)),
    newspot_y = ifelse(flip_direction == "counterclockwise", -spot_x,spot_x),
    newspot_x = ifelse(flip_direction == "counterclockwise",spot_y, -spot_y + 53.33),
    line_to_gain = newspot_y - yardsToGo
  ) %>% 
  mutate(across(
    c(newy,newspot_y,line_to_gain),
    ~ifelse(flip_direction == "counterclockwise",.x + 110,.x - 10)
  )) %>% 
  group_by(gameId,playId,nflId,jerseyNumber) %>% 
  list(
    distinct(.,newspot_y,newspot_x,line_to_gain) %>% 
      ungroup,
    summarise(
      .,
      across(c(newx,newy),mean),
      .groups = "drop"
    )
  ) %>% 
  .[-1] %>% 
  reduce(left_join,by = c("gameId","playId","nflId","jerseyNumber")) %>% 
  collect

depth_by_play <- mean_positions %>% 
  inner_join(
    plays_clean %>% 
      filter(pff_passCoverage %in% coverages_to_model) %>% 
      select(gameId,playId,pff_passCoverage),
    by = c("gameId","playId")
  ) %>% 
  mutate(
    depth = newy - newspot_y,
    width = newx - newspot_x,
    first_down = line_to_gain - newspot_y
  ) %>% 
  group_by(gameId,playId) %>% 
  mutate(
    width_rk = data.table::frankv(width,order = -1,ties.method = "min"),
    side = ifelse(width < 0,"L","R"),
    depth_rk = data.table::frankv(depth,order = 1,ties.method = "min")
  ) %>% 
  ungroup

depth_by_play %>% 
  list(
    distinct(.,gameId,playId) %>% 
      slice_sample(n = 1)
  ) %>% 
  reduce(inner_join,by = c("gameId","playId")) %>% {
    ggplot(.,mapping = aes(x = width,y = depth)) +
      geom_hline(yintercept = unique(.$first_down))+
      geom_hline(yintercept = 0,lty = 3)+
      geom_vline(xintercept = 0,lty = 3)+
      geom_point(size= 8,color = "black")+
      geom_text(aes(label = depth_rk,size = 4),color = "white")+
      ggtitle(label = unique(.$pff_passCoverage))+
      theme(theme_classic(),axis.text.x = element_text(size = 10),
            legend.position = "none")
  }
  
depth_by_play %>% 
  list(
    filter(.,corner == "non-corner") %>% 
      group_by(pff_passCoverage,side,depth_rk) %>% 
      summarise(across(c(width,depth),mean),
                .groups = "drop")
  ) %>% 
  .[[2]] %>% 
  filter(pff_passCoverage == "Cover-2")
  ggplot(mapping = aes(x = width,y = depth))+
  geom_hline(yintercept = 0,lty = 3)+
  geom_vline(xintercept = 0,lty = 3)+
  geom_point(size= 8)+
  geom_text(aes(label = depth_rk,size = 4),color = "white")+
  facet_wrap(vars(pff_passCoverage,side))
  
depth_by_play %>% 
  filter(depth_rk >= 5) %>% 
  group_by(pff_passCoverage,width_rk) %>% 
  summarise(across(c(width,depth),mean),
            .groups = "drop") %>% 
  ggplot(mapping = aes(x = width,y = depth))+
  geom_hline(yintercept = 0,lty = 3)+
  geom_vline(xintercept = 0,lty = 3)+
  geom_point(size= 8)+
  geom_text(aes(label = width_rk,size = 4),color = "white")+
  theme(axis.text.x = element_text(size = 10))+
  facet_wrap(vars(pff_passCoverage))

#### add zones ####

cover2 <- tibble(
  x = c(-15,15,-10,10,0),
  y = c(-7.5,-7.5,-15,-15,-8.75),
  w = c(10,10,20,20,20),
  h = c(5,5,10,10,7.5),
  z = c("flat","flat","half","half","hooks/pole")
)

cover2_plot <- 
  geom_tile(
    data = cover2,
    mapping = aes(x = x,y = y,width = w,height = h,fill = z),color = "black"
  )

### this is the informative graph right here
depth_by_play %>% 
  filter(depth_rk <= 7) %>% 
  group_by(pff_passCoverage,width_rk,depth_rk) %>% 
  summarise(count = n(),
            across(c(width,depth,first_down),mean),
            .groups = "drop") %>%
  mutate(pos = paste(depth_rk,width_rk,sep = "-")) %>% 
  ggplot(mapping = aes(x = width,y = depth))+
  cover2_plot + 
  geom_hline(yintercept = 0,lty = 3)+
  geom_vline(xintercept = 0,lty = 3)+
  geom_vline(xintercept = -3.081667,lty = 2)+
  geom_vline(xintercept = 3.081667,lty = 2)+
  geom_point(aes(size = count)) +
  theme(axis.text.x = element_text(size = 10))+
  facet_wrap(vars(pff_passCoverage))

depth_by_play %>% 
  filter(depth_rk %in% c(1,2)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    .groups = "drop"
  ) %>% 
  #filter()
  ggplot(mapping = aes(x = depth_diff,fill = pff_passCoverage))+
  geom_density(alpha = .6)+
  theme(axis.text.x = element_text(size = 10))
  
depth_by_play %>% 
  filter(depth_rk %in% c(1,2)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    max_width = max(abs(width)),
    .groups = "drop"
  ) %>% 
  group_by(pff_passCoverage) %>% 
  summarise(
    count = n(),
    across(c(depth_diff,max_width),mean),
    .groups = "drop"
  ) %>% 
  ggplot(mapping = aes(x = max_width,y = depth_diff))+
  geom_point(aes(size = count,color = pff_passCoverage))+
  geom_label_repel(aes(label = pff_passCoverage),color = "black")+
  scale_size_continuous(range = c(2,10))+
  theme(axis.text.x = element_text(size = 10),legend.position = "none")

depth_by_play %>% 
  filter(depth_rk %in% c(1,2)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    max_width = max(abs(width)),
    .groups = "drop"
  ) %>% 
  ggplot(mapping = aes(x = max_width,y = depth_diff))+
  geom_point(aes(color = pff_passCoverage))+
  theme(axis.text.x = element_text(size = 10))+
  scale_x_continuous(limits = c(0,25))

deepest_players <- depth_by_play %>% 
  filter(depth_rk %in% c(1,2)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    max_width = max(abs(width)),
    .groups = "drop"
  ) 

require(tidymodels)
require(class)

#### model ####

set.seed(15)
model_split <- initial_split(deepest_players,prop = .8)
model_train <- training(model_split)
train_split <- initial_split(model_train,prop = .75)
model_ana <- training(train_split)
model_assess <- testing(train_split)
model_test <- testing(model_split)

train_y <- model_ana$pff_passCoverage

assess_pred <- knn(
  train = model_ana %>% select(depth_diff,max_width),
  test = model_assess %>% select(depth_diff,max_width),
  cl = train_y,
  k = 10
)

all_preds <- list(
  model_assess %>% 
    select(depth_diff,max_width),
  grid <- crossing(
    depth_diff = seq(0,20,by = 1),
    max_width = seq(0,20,by = 1)
  )
) %>% 
  map(function(x) {
    
    knn(
      train = model_ana %>% select(depth_diff,max_width),
      test = x,
      cl = train_y,
      k = 30
    )
    
  })

ktune <- function(model_ana, test_df, grid, k, train_y) {
  
  preds <- list(
    test_df,
    grid
  ) %>% 
    map(function(x) {
      
      knn(
        train = model_ana %>% select(depth_diff,max_width),
        test = x,
        cl = train_y,
        k = k
      )
      
    })
  
  return(preds)
  
}

k_grid <- seq(10,200,by = 10) %>% 
  tibble(k = .) %>% 
  mutate(knn = map(k,function(k) {
    
    ktune(
      model_ana,
      model_assess %>% 
        select(depth_diff,max_width),
      grid <- crossing(
        depth_diff = seq(0,20,by = 1),
        max_width = seq(0,20,by = 1)
      ),
      k = k
    )
    
  }))

k_grid %>% 
  mutate(assess_preds = map(knn,~.x[1])) %>% 
  select(-knn) %>% 
  unnest(cols = assess_preds) %>% 
  mutate(actual = list(model_assess$pff_passCoverage)) %>% 
  unnest(cols = c(assess_preds,actual)) %>% 
  group_by(k) %>% 
  summarise(accuracy = sum(assess_preds == actual)/n()) %>% 
  ggplot(mapping = aes(x = k, y = accuracy))+
  geom_point()+
  geom_path()+
  theme(axis.text.x = element_text(size = 10))

k_best <- k_grid %>% 
  filter(k == 50) %>% 
  mutate(assess_preds = map(knn,~.x[1]),
         assess_grid = map(knn,~.x[2])) %>% 
  select(-knn)

model_assess %>% 
  mutate(pred = k_best$assess_preds[[1]][[1]]) %>% 
  count(pff_passCoverage,pred) %>% 
  mutate(match = (pff_passCoverage == pred)) %>% 
  group_by(pff_passCoverage) %>% 
  summarise(
    count = sum(n),
    correct = sum(n[match == T]),
    .groups = "drop"
  ) %>% 
  mutate(pct = correct/count)

model_assess %>% 
  mutate(pred = k_best$assess_preds[[1]][[1]]) %>% 
  count(pff_passCoverage,pred) %>% 
  group_by(pff_passCoverage) %>% 
  mutate(pct = n/sum(n)) %>% 
  ungroup %>% 
  pivot_wider(id_cols = pff_passCoverage,names_from = pred,values_from = pct)

grid %>% 
  mutate(
    pred = k_best$assess_grid[[1]][[1]],
    pred_bin = cut(as.numeric(pred), breaks = seq(0.5, 6.5, by = 1), include.lowest = TRUE, labels = names(color_list)) 
  ) %>% {
  ggplot(.)+
  geom_contour_filled(
    aes(x = max_width, y = depth_diff, z = as.numeric(pred)),
    breaks = seq(0.5,6.5,by = 1),
    color = "black"
  )+
  scale_fill_discrete(labels = levels(.$pred))+
  theme_minimal()+
  labs(fill = "")
  }

#### 1-high or not ####

train_hierarchy <- model_ana %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  pull(hierarchy)

assess_hierarchy <- model_assess %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  pull(hierarchy)

ktune <- function(model_ana, test_df, grid, k, train_y) {
  
  preds <- list(
    test_df,
    grid
  ) %>% 
    map(function(x) {
      
      knn(
        train = model_ana %>% select(depth_diff,max_width),
        test = x,
        cl = train_y,
        k = k
      )
      
    })
  
  return(preds)
  
}

k_grid <- seq(10,200,by = 10) %>% 
  tibble(k = .) %>% 
  mutate(knn = map(k,function(k) {
    
    ktune(
      model_ana,
      model_assess %>% 
        select(depth_diff,max_width),
      grid <- crossing(
        depth_diff = seq(0,20,by = 1),
        max_width = seq(0,20,by = 1)
      ),
      k = k,
      train_y = train_hierarchy
    )
    
  }))

k_grid %>% 
  mutate(assess_preds = map(knn,~.x[1])) %>% 
  select(-knn) %>% 
  unnest(cols = assess_preds) %>% 
  mutate(actual = list(assess_hierarchy)) %>% 
  unnest(cols = c(assess_preds,actual)) %>% 
  group_by(k) %>% 
  summarise(accuracy = sum(assess_preds == actual)/n()) %>% 
  ggplot(mapping = aes(x = k, y = accuracy))+
  geom_point()+
  geom_path()+
  theme(axis.text.x = element_text(size = 10))

k_best <- k_grid %>% 
  filter(k == 60) %>% 
  mutate(assess_preds = map(knn,~.x[1]),
         assess_grid = map(knn,~.x[2])) %>% 
  select(-knn)

model_assess %>% 
  mutate(pred = k_best$assess_preds[[1]][[1]],
         actual = assess_hierarchy) %>% 
  count(actual,pred) %>% 
  mutate(match = (actual == pred)) %>% 
  group_by(actual) %>% 
  summarise(
    count = sum(n),
    correct = sum(n[match == T]),
    .groups = "drop"
  ) %>% 
  mutate(pct = correct/count)

model_assess %>% 
  mutate(pred = k_best$assess_preds[[1]][[1]],
         actual = assess_hierarchy) %>% 
  count(actual,pred) %>% 
  group_by(actual) %>% 
  mutate(pct = n/sum(n)) %>% 
  ungroup %>% 
  pivot_wider(id_cols = actual,names_from = pred,values_from = pct)

grid %>% 
  mutate(
    pred = k_best$assess_grid[[1]][[1]],
    pred_bin = cut(as.numeric(pred), breaks = seq(0.5, 6.5, by = 1), include.lowest = TRUE, labels = names(color_list)) 
  ) %>% {
    ggplot(.)+
      geom_contour_filled(
        aes(x = max_width, y = depth_diff, z = as.numeric(pred)),
        breaks = seq(0.5,6.5,by = 1),
        color = "black"
      )+
      scale_fill_discrete(labels = levels(.$pred))+
      theme_minimal()+
      labs(fill = "")
  }

#### cv ####

k_grid <- deepest_players %>% 
  semi_join(model_train,by = c('gameId','playId')) %>% 
  mutate(across(c(depth_diff,max_width),~scale(.)[,1])) %>% 
  mutate(
    index = sample(nrow(.)),
    fold = cut(index,breaks = seq(1,nrow(.),length.out = 6),include.lowest = T) %>% as.integer,
    hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)
  ) %>% 
  select(fold,depth_diff,max_width,hierarchy) %>% 
  crossing(kfold = seq(1,5,by = 1)) %>% 
  nest(whole = c(fold,depth_diff,max_width,hierarchy)) %>% 
  mutate(
    ana = pmap(list(whole,kfold), function(x,y) filter(x,fold != y)),
    assess = pmap(list(whole,kfold), function(x,y) filter(x,fold == y))
  ) %>% 
  crossing(k = seq(10,150,by = 10)) %>% 
  mutate(
    knn = pmap(list(ana,assess,k), function(x,y,z) {
      
      knn(
        train = select(x,depth_diff,max_width),
        test = select(y,depth_diff,max_width),
        cl = x$hierarchy,
        k = z
      )
      
    })
  )

k_grid %>% 
  select(k,assess,knn) %>% 
  unnest(cols = c(assess,knn)) %>% 
  group_by(k) %>% 
  summarise(
    count = n(),
    correct = sum(hierarchy == knn)
  ) %>% 
  mutate(accuracy = correct/count)

test_preds <- model_train %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width),
      test = model_test %>% select(depth_diff,max_width),
      cl = .$hierarchy,
      k = 150
    )
  )

grid_preds <- model_train %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width),
      test = grid,
      cl = .$hierarchy,
      k = 150
    )
  ) %>% 
  .[[2]]

grid %>% 
  mutate(pred = grid_preds) %>% {
    ggplot(.)+
      geom_contour_filled(
        aes(x = max_width, y = depth_diff, z = as.numeric(pred)),
        breaks = seq(0.5,6.5,by = 1),
        color = "black"
      )+
      scale_fill_discrete(labels = levels(.$pred))+
      theme_minimal()+
      labs(fill = "")
  }

#### third dimension ####

deepest_players3d <- depth_by_play %>% 
  mutate(centerfield = depth <= -15 & abs(width) < 5) %>% 
  filter(depth_rk %in% c(1,2)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    max_width = max(abs(width)),
    centerfielders = sum(centerfield),
    .groups = "drop"
  ) 

k_grid <- deepest_players3d %>% 
  semi_join(model_train,by = c('gameId','playId')) %>% 
  mutate(across(c(depth_diff,max_width),~scale(.)[,1])) %>% 
  mutate(
    index = sample(nrow(.)),
    fold = cut(index,breaks = seq(1,nrow(.),length.out = 6),include.lowest = T) %>% as.integer,
    hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)
  ) %>% 
  select(fold,depth_diff,max_width,hierarchy,centerfielders) %>% 
  crossing(kfold = seq(1,5,by = 1)) %>% 
  nest(whole = c(fold,depth_diff,max_width,centerfielders,hierarchy)) %>% 
  mutate(
    ana = pmap(list(whole,kfold), function(x,y) filter(x,fold != y)),
    assess = pmap(list(whole,kfold), function(x,y) filter(x,fold == y))
  ) %>% 
  crossing(k = seq(10,150,by = 10)) %>% 
  mutate(
    knn = pmap(list(ana,assess,k), function(x,y,z) {
      
      knn(
        train = select(x,depth_diff,max_width,centerfielders),
        test = select(y,depth_diff,max_width,centerfielders),
        cl = x$hierarchy,
        k = z
      )
      
    })
  )

k_grid %>% 
  select(k,assess,knn) %>% 
  unnest(cols = c(assess,knn)) %>% 
  group_by(k) %>% 
  summarise(
    count = n(),
    correct = sum(hierarchy == knn)
  ) %>% 
  mutate(accuracy = correct/count)

test_preds <- model_train %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width),
      test = model_test %>% select(depth_diff,max_width),
      cl = .$hierarchy,
      k = 100
    )
  )

grid_preds <- deepest_players3d %>% 
  semi_join(model_train,by = c('gameId','playId')) %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width,centerfielders),
      test = grid %>% crossing(centerfielders = c(0,1,2)),
      cl = .$hierarchy,
      k = 150
    )
  ) %>% 
  .[[2]]

grid %>% 
  crossing(centerfielders = c(0,1,2)) %>% 
  mutate(pred = grid_preds) %>% 
  {
    ggplot(.)+
      geom_contour_filled(
        aes(x = max_width, y = depth_diff, z = as.numeric(pred)),
        breaks = seq(0.5,6.5,by = 1),
        color = "black"
      )+
      scale_fill_discrete(labels = levels(.$pred))+
      theme_minimal()+
      labs(fill = "")+
      facet_wrap(vars(centerfielders))
  }

#### third player? ####

deepest_players <- depth_by_play %>% 
  filter(depth_rk %in% c(1,2,3)) %>% 
  group_by(gameId,playId,pff_passCoverage) %>% 
  summarise(
    depth_diff = max(depth) - min(depth),
    max_width = max(abs(width)),
    wide_width = mean(width[width != min(width)]),
    .groups = "drop"
  ) 

k_grid <- deepest_players %>% 
  semi_join(model_train,by = c('gameId','playId')) %>% 
  mutate(across(c(depth_diff,max_width,wide_width),~scale(.)[,1])) %>% 
  mutate(
    index = sample(nrow(.)),
    fold = cut(index,breaks = seq(1,nrow(.),length.out = 6),include.lowest = T) %>% as.integer,
    hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)
  ) %>% 
  select(fold,depth_diff,max_width,hierarchy,wide_width) %>% 
  crossing(kfold = seq(1,5,by = 1)) %>% 
  nest(whole = c(fold,depth_diff,max_width,wide_width,hierarchy)) %>% 
  mutate(
    ana = pmap(list(whole,kfold), function(x,y) filter(x,fold != y)),
    assess = pmap(list(whole,kfold), function(x,y) filter(x,fold == y))
  ) %>% 
  crossing(k = seq(10,150,by = 10)) %>% 
  mutate(
    knn = pmap(list(ana,assess,k), function(x,y,z) {
      
      knn(
        train = select(x,depth_diff,max_width,wide_width),
        test = select(y,depth_diff,max_width,wide_width),
        cl = x$hierarchy,
        k = z
      )
      
    })
  )

k_grid %>% 
  select(k,assess,knn) %>% 
  unnest(cols = c(assess,knn)) %>% 
  group_by(k) %>% 
  summarise(
    count = n(),
    correct = sum(hierarchy == knn)
  ) %>% 
  mutate(accuracy = correct/count)

test_preds <- model_train %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width),
      test = model_test %>% select(depth_diff,max_width),
      cl = .$hierarchy,
      k = 100
    )
  )

grid_preds <- deepest_players3d %>% 
  semi_join(model_train,by = c('gameId','playId')) %>% 
  mutate(hierarchy = ifelse(pff_passCoverage %in% c("Cover-1","Cover-3"),1,0)) %>% 
  list(
    knn(
      train = select(.,depth_diff,max_width,centerfielders),
      test = grid %>% crossing(centerfielders = c(0,1,2)),
      cl = .$hierarchy,
      k = 150
    )
  ) %>% 
  .[[2]]

grid %>% 
  crossing(centerfielders = c(0,1,2)) %>% 
  mutate(pred = grid_preds) %>% 
  {
    ggplot(.)+
      geom_contour_filled(
        aes(x = max_width, y = depth_diff, z = as.numeric(pred)),
        breaks = seq(0.5,6.5,by = 1),
        color = "black"
      )+
      scale_fill_discrete(labels = levels(.$pred))+
      theme_minimal()+
      labs(fill = "")+
      facet_wrap(vars(centerfielders))
  }
