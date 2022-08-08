use std::sync::{Arc, Mutex};
use druid::{Affine, BoxConstraints, Color, Data, Env, Event,
            EventCtx, LayoutCtx, LifeCycle, LifeCycleCtx, PaintCtx,
            piet, Point, RenderContext, Size, UpdateCtx, Widget};
use druid::kurbo::{BezPath, PathEl};
use druid::piet::{Device, ImageFormat, LineCap, LineJoin, PaintBrush, StrokeStyle};

const TRANSPARENT: Color = Color::rgba8(0, 0, 0, 0xff);

struct CanvasForeground {
    path: BezPath,
    stroke_brush: PaintBrush,
    stroke_width: f64,
}

impl Default for CanvasForeground {
    fn default() -> Self {
        CanvasForeground {
            path: BezPath::new(),
            stroke_brush: TRANSPARENT.into(),
            stroke_width: 0.0
        }
    }
}

impl CanvasForeground {
    fn extend_stroke_path(&mut self, size: Size, mut next_point: Point) {
        next_point.x /= size.width;
        next_point.y /= size.height;

        self.path.push(PathEl::LineTo(next_point))
    }

    fn begin_stroke_path(&mut self, size: Size, mut starting_point: Point) {
        starting_point.x /= size.width;
        starting_point.y /= size.height;

        self.path.push(PathEl::MoveTo(starting_point))
    }

    fn draw_stroke(&self, size: Size, ctx: &mut impl RenderContext) {
        ctx.with_save(|ctx| {
            ctx.transform(Affine::scale_non_uniform(size.width, size.height));
            ctx.stroke_styled(&self.path,
                              &self.stroke_brush,
                              self.stroke_width,
                              &StrokeStyle::new()
                                  .line_cap(LineCap::Round)
                                  .line_join(LineJoin::Round));

            Ok(())
        }).unwrap();
    }

    fn clear(&mut self) {
        self.path = BezPath::new();
    }
}

struct CanvasContent {
    background: PaintBrush,
    foreground: CanvasForeground,
}

impl Default for CanvasContent {
    fn default() -> Self {
        CanvasContent {
            background: TRANSPARENT.into(),
            foreground: CanvasForeground::default(),
        }
    }
}

impl CanvasContent {
    fn draw(&self, size: Size, ctx: &mut impl RenderContext) {
        ctx.fill(size.to_rect(), &self.background);
        self.foreground.draw_stroke(size, ctx);
    }

    fn clear(&mut self) {
        self.foreground.clear();
    }
}

#[derive(Default)]
struct MousePositionTracker {
    old_pos: Point,
    pos: Point,
    is_down: bool
}

impl MousePositionTracker {
    fn mouse_down(&mut self, pos: Point) {
        self.is_down = true;
        self.old_pos = pos;
        self.pos = pos;
    }

    fn mouse_move(&mut self, new_pos: Point) {
        self.old_pos = self.pos;
        self.pos = new_pos;
    }

    fn mouse_up(&mut self) {
        self.is_down = false;
    }
}

pub struct InteractiveCanvasStateBuilder {
    content: CanvasContent
}

impl InteractiveCanvasStateBuilder {
    fn new() -> InteractiveCanvasStateBuilder {
        InteractiveCanvasStateBuilder {
            content: CanvasContent::default()
        }
    }

    pub fn with_background(mut self, background: impl Into<PaintBrush>) -> Self {
        self.content.background = background.into();
        self
    }

    pub fn with_stroke_brush(mut self, stroke_brush: impl Into<PaintBrush>) -> Self {
        self.content.foreground.stroke_brush = stroke_brush.into();
        self
    }

    pub fn with_stroke_width(mut self, stroke_width: f64) -> Self {
        self.content.foreground.stroke_width = stroke_width.into();
        self
    }

    pub fn build(self) -> InteractiveCanvasState {
        InteractiveCanvasState {
            content: Arc::new(self.content.into()),
            flag: false
        }
    }
}

#[derive(Data, Clone)]
pub struct InteractiveCanvasState {
    content: Arc<Mutex<CanvasContent>>,
    flag: bool
}

impl InteractiveCanvasState {
    pub fn builder() -> InteractiveCanvasStateBuilder {
        InteractiveCanvasStateBuilder::new()
    }

    pub fn clear(&mut self) {
        self.content.lock().unwrap().clear();
        self.invalidate();
    }

    pub fn copy_pixels(&self, size_dimension: u32) -> Result<Vec<u8>, piet::Error> {
        let content = self.content.lock().unwrap();
        let mut device = Device::new()?;
        let mut target =
            device
                .bitmap_target(size_dimension as usize,
                               size_dimension as usize,
                               size_dimension as f64)?;

        let size = Size::new(size_dimension as f64,
                             size_dimension as f64);

        let mut context = target.render_context();
        content.draw(size,&mut context);

        context.finish()?;

        let mut buf = vec![0u8; size.area() as usize];
        target.copy_raw_pixels(ImageFormat::RgbaPremul, &mut buf)?;

        Ok(buf)
    }

    fn invalidate(&mut self) {
        self.flag = !self.flag;
    }
}

impl Default for InteractiveCanvasState {
    fn default() -> Self {
        InteractiveCanvasState::builder().build()
    }
}

pub type BoxedStateProvider<T> = Box<dyn Fn(&T) -> InteractiveCanvasState>;

pub struct InteractiveCanvas<T> {
    mouse_tracker: MousePositionTracker,
    state_provider: BoxedStateProvider<T>
}

impl<T> Default for InteractiveCanvas<T> {
    fn default() -> Self {
        InteractiveCanvas {
            mouse_tracker: MousePositionTracker::default(),
            state_provider: Box::new(|_| InteractiveCanvasState::default())
        }
    }
}

impl<T> InteractiveCanvas<T> {
    pub fn with_state<F>(mut self, get_state: F) -> Self
        where
            F: (Fn(&T) -> InteractiveCanvasState) + 'static
    {
        self.state_provider = Box::new(get_state);
        self
    }
}

impl<T: Data> Widget<T> for InteractiveCanvas<T> {
    fn event(&mut self,
             ctx: &mut EventCtx,
             event: &Event,
             data: &mut T,
             _env: &Env) {
        let state = (self.state_provider)(data);
        match event {
            Event::MouseDown(event) => {
                self.mouse_tracker.mouse_down(event.pos);
                state.content.lock().unwrap().foreground
                    .begin_stroke_path(ctx.size(), self.mouse_tracker.pos);
            },
            Event::MouseMove(event) => {
                self.mouse_tracker.mouse_move(event.pos);
                if self.mouse_tracker.is_down {
                    state.content.lock().unwrap().foreground
                        .extend_stroke_path(ctx.size(), self.mouse_tracker.pos);

                    ctx.request_paint();
                }
            },
            Event::MouseUp(_) => self.mouse_tracker.mouse_up(),
            _ => {}
        }
    }

    fn lifecycle(&mut self,
                 _ctx: &mut LifeCycleCtx,
                 _event: &LifeCycle,
                 _data: &T,
                 _env: &Env) {}

    fn update(&mut self,
              ctx: &mut UpdateCtx,
              _old_data: &T,
              _data: &T,
              _env: &Env) {
        ctx.request_paint();
    }

    fn layout(&mut self,
              _ctx: &mut LayoutCtx,
              bc: &BoxConstraints,
              _data: &T,
              _env: &Env) -> Size {
        bc.max()
    }

    fn paint(&mut self, ctx: &mut PaintCtx, data: &T, _env: &Env) {
        (self.state_provider)(data).content.lock().unwrap()
            .draw(ctx.region().bounding_box().size(), ctx.render_ctx)
    }
}
